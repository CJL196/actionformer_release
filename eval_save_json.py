# python imports
import argparse
import os
import glob
import time
import json
from pprint import pprint
from collections import OrderedDict

# torch imports
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.utils.data

# our code
from libs.core import load_config
from libs.datasets import make_dataset, make_data_loader
from libs.modeling import make_meta_arch
from libs.utils import fix_random_seed, AverageMeter, batched_nms


def valid_one_epoch_save_json(
    val_loader,
    model,
    output_file,
    cfg,
    print_freq=20
):
    """Test the model on the validation set and save the results in a JSON file."""
    # set up meters
    batch_time = AverageMeter()
    # switch to evaluate mode
    model.eval()

    # dict for results
    all_results = {}
    
    # test cfg
    test_cfg = cfg['test_cfg']

    # loop over validation set
    start = time.time()
    for iter_idx, video_list in enumerate(val_loader, 0):
        # forward the model (wo. grad)
        with torch.no_grad():
            output = model(video_list)

            # unpack the results
            num_vids = len(output)
            for vid_idx in range(num_vids):
                video_id = output[vid_idx]['video_id']
                if output[vid_idx]['segments'].shape[0] > 0:
                    # parse video number
                    vid_num_str = ''.join(filter(str.isdigit, video_id))
                    vid_num = int(vid_num_str) if vid_num_str else 0

                    segments = output[vid_idx]['segments'].cpu()
                    labels = output[vid_idx]['labels'].cpu()
                    scores = output[vid_idx]['scores'].cpu()

                    if segments.shape[0] > 0:
                        # multiclass NMS, results are sorted by score
                        segments, scores, labels = batched_nms(
                            segments,
                            scores,
                            labels,
                            test_cfg['iou_threshold'],
                            test_cfg['min_score'],
                            test_cfg['max_seg_num'],
                            use_soft_nms=True if test_cfg['nms_method'] == 'soft' else False,
                            multiclass=test_cfg['multiclass_nms'],
                            sigma=test_cfg['nms_sigma'],
                            voting_thresh=test_cfg['voting_thresh']
                        )

                    # Truncate the number of segments based on video ID
                    max_segments = 7 if vid_num <= 728 else 12
                    
                    num_to_keep = min(len(segments), max_segments)
                    segments = segments[:num_to_keep]
                    scores = scores[:num_to_keep]
                    labels = labels[:num_to_keep]

                    results_for_video = []
                    for seg, score, lbl in zip(segments, scores, labels):
                        results_for_video.append({
                            'segment': seg.tolist(),
                            'label': lbl.item(),
                            'score': score.item()
                        })
                            
                    # sort segments by start time
                    results_for_video.sort(key=lambda x: x['segment'][0])
                    
                    all_results[video_id] = results_for_video

        # printing
        if (iter_idx != 0) and iter_idx % (print_freq) == 0:
            # measure elapsed time (sync all kernels)
            torch.cuda.synchronize()
            batch_time.update((time.time() - start) / print_freq)
            start = time.time()

            # print timing
            print('Test: [{0:05d}/{1:05d}]\t'
                  'Time {batch_time.val:.2f} ({batch_time.avg:.2f})'.format(
                  iter_idx, len(val_loader), batch_time=batch_time))

    # sort videos by id
    sorted_all_results = OrderedDict(sorted(all_results.items(), key=lambda x: int(''.join(filter(str.isdigit, x[0])))))


    # save to json
    with open(output_file, 'w') as f:
        json.dump(sorted_all_results, f, indent=4)

    return


################################################################################
def main(args):
    """0. load config"""
    # sanity check
    if os.path.isfile(args.config):
        cfg = load_config(args.config)
    else:
        raise ValueError("Config file does not exist.")
    assert len(cfg['val_split']) > 0, "Test set must be specified!"
    if ".pth.tar" in args.ckpt:
        assert os.path.isfile(args.ckpt), "CKPT file does not exist!"
        ckpt_file = args.ckpt
    else:
        assert os.path.isdir(args.ckpt), "CKPT file folder does not exist!"
        if args.epoch > 0:
            ckpt_file = os.path.join(
                args.ckpt, 'epoch_{:03d}.pth.tar'.format(args.epoch)
            )
        else:
            ckpt_file_list = sorted(glob.glob(os.path.join(args.ckpt, '*.pth.tar')))
            ckpt_file = ckpt_file_list[-1]
        assert os.path.exists(ckpt_file)

    if args.topk > 0:
        cfg['model']['test_cfg']['max_seg_num'] = args.topk
    pprint(cfg)

    """1. fix all randomness"""
    # fix the random seeds (this will fix everything)
    _ = fix_random_seed(0, include_cuda=True)

    """2. create dataset / dataloader"""
    val_dataset = make_dataset(
        cfg['dataset_name'], False, cfg['val_split'] + cfg['train_split'], **cfg['dataset']
    )
    # set bs = 1, and disable shuffle
    val_loader = make_data_loader(
        val_dataset, False, None, 1, cfg['loader']['num_workers']
    )

    """3. create model and evaluator"""
    # model
    model = make_meta_arch(cfg['model_name'], **cfg['model'])
    # not ideal for multi GPU training, ok for now
    model = nn.DataParallel(model, device_ids=cfg['devices'])

    """4. load ckpt"""
    print("=> loading checkpoint '{}'".format(ckpt_file))
    # load ckpt, reset epoch / best rmse
    checkpoint = torch.load(
        ckpt_file,
        map_location = lambda storage, loc: storage.cuda(cfg['devices'][0])
    )
    # load ema model instead
    print("Loading from EMA model ...")
    model.load_state_dict(checkpoint['state_dict_ema'])
    del checkpoint

    # set up output file
    output_file = os.path.join(os.path.split(ckpt_file)[0], 'eval_results.json')

    """5. Test the model"""
    print("\nStart testing model {:s} ...".format(cfg['model_name']))
    start = time.time()

    valid_one_epoch_save_json(
        val_loader,
        model,
        output_file,
        cfg,
        print_freq=args.print_freq
    )

    end = time.time()
    print("All done! Total time: {:0.2f} sec".format(end - start))
    print("Results saved to {:s}".format(output_file))
    return

################################################################################
if __name__ == '__main__':
    """Entry Point"""
    # the arg parser
    parser = argparse.ArgumentParser(
      description='Train a point-based transformer for action localization')
    parser.add_argument('config', type=str, metavar='DIR',
                        help='path to a config file')
    parser.add_argument('ckpt', type=str, metavar='DIR',
                        help='path to a checkpoint')
    parser.add_argument('-epoch', type=int, default=-1,
                        help='checkpoint epoch')
    parser.add_argument('-t', '--topk', default=-1, type=int,
                        help='max number of output actions (default: -1)')
    parser.add_argument('-p', '--print-freq', default=10, type=int,
                        help='print frequency (default: 10 iterations)')
    args = parser.parse_args()
    main(args) 