import os, cv2, sys

import time
import collections
import torch
import argparse
import numpy as np
import util
import models
from pypse import pse as pypse
from torch.autograd import Variable
from dataset import IC15TestLoader
import Polygon as plg


def extend_3c(img):
    img = img.reshape(img.shape[0], img.shape[1], 1)
    img = np.concatenate((img, img, img), axis=2)
    return img


def debug(idx, img_paths, imgs, output_root):
    if not os.path.exists(output_root):
        os.makedirs(output_root)

    col = []
    for i in range(len(imgs)):
        row = []
        for j in range(len(imgs[i])):
            # img = cv2.copyMakeBorder(imgs[i][j], 3, 3, 3, 3, cv2.BORDER_CONSTANT, value=[255, 0, 0])
            row.append(imgs[i][j])
        res = np.concatenate(row, axis=1)
        col.append(res)
    res = np.concatenate(col, axis=0)
    img_name = img_paths[idx].split('/')[-1]
    print(idx + 1, '/', len(img_paths), img_name)
    cv2.imwrite(output_root + img_name, res)


def get_coords_and_write(image_name, bboxes, path):
    '''
     return x1,y1,x4,y4
    :param image_name:
    :param bboxes:
    :param path:
    :return:
    '''
    filename = util.io.join_path(path, 'res_%s.txt' % (image_name))
    lines = []
    for b_idx, bbox in enumerate(bboxes):
        values = [int(v) for v in bbox]
        line = "%d, %d, %d, %d, %d, %d, %d, %d\n" % tuple(values)
        lines.append(line)
    new_lines = fliter_coords(lines)
    util.io.write_lines(filename, str(new_lines))
    return new_lines


def fliter_coords(point_list):
    '''
    return x1,y1,x4,y4
    :param point_list:
    :return:
    '''

    result = []
    for old_point in point_list:
        new_list = []
        old_point = old_point.replace('\n', '').split(',')
        x = [int(old_point[0]), int(old_point[2]), int(old_point[4]), int(old_point[6])]
        y = [int(old_point[1]), int(old_point[3]), int(old_point[5]), int(old_point[7])]
        new_list.append(min(x))
        new_list.append(min(y))
        new_list.append(max(x) + 5)
        new_list.append(max(y) + 1)
        result.append(new_list)
    return result


def polygon_from_points(points):
    """
    Returns a Polygon object to use with the Polygon2 class from a list of 8 points: x1,y1,x2,y2,x3,y3,x4,y4
    """
    resBoxes = np.empty([1, 8], dtype='int32')
    resBoxes[0, 0] = int(points[0])
    resBoxes[0, 4] = int(points[1])
    resBoxes[0, 1] = int(points[2])
    resBoxes[0, 5] = int(points[3])
    resBoxes[0, 2] = int(points[4])
    resBoxes[0, 6] = int(points[5])
    resBoxes[0, 3] = int(points[6])
    resBoxes[0, 7] = int(points[7])
    pointMat = resBoxes[0].reshape([2, 4]).T
    return plg.Polygon(pointMat)


def test(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.CUDA
    data_loader = IC15TestLoader(long_size=args.long_size, validate_data_path=args.validate_data_path)
    test_loader = torch.utils.data.DataLoader(
        data_loader,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        drop_last=True)

    # Setup Model
    if args.arch == "resnet50":
        model = models.resnet50(pretrained=True, num_classes=7, scale=args.scale)
    elif args.arch == "resnet101":
        model = models.resnet101(pretrained=True, num_classes=7, scale=args.scale)
    elif args.arch == "resnet152":
        model = models.resnet152(pretrained=True, num_classes=7, scale=args.scale)

    for param in model.parameters():
        param.requires_grad = False

    model = model.cuda()

    if args.model_path is not None:
        if os.path.isfile(args.model_path):
            print("Loading model and optimizer from checkpoint '{}'".format(args.model_path))
            checkpoint = torch.load(args.model_path)

            # model.load_state_dict(checkpoint['state_dict'])
            d = collections.OrderedDict()
            for key, value in checkpoint['state_dict'].items():
                tmp = key[7:]
                d[tmp] = value
            model.load_state_dict(d)

            print("Loaded checkpoint '{}' (epoch {})"
                  .format(args.model_path, checkpoint['epoch']))
            sys.stdout.flush()
        else:
            print("No checkpoint found at '{}'".format(args.model_path))
            sys.stdout.flush()

    model.eval()

    total_frame = 0.0
    total_time = 0.0
    for idx, (org_img, img) in enumerate(test_loader):
        print('progress: %d / %d' % (idx, len(test_loader)))
        sys.stdout.flush()
        img = Variable(img.cuda())
        org_img = org_img.numpy().astype('uint8')[0]
        text_box = org_img.copy()

        torch.cuda.synchronize()
        start = time.time()

        outputs = model(img)

        score = torch.sigmoid(outputs[:, 0, :, :])
        outputs = (torch.sign(outputs - args.binary_th) + 1) / 2

        text = outputs[:, 0, :, :]
        kernels = outputs[:, 0:args.kernel_num, :, :] * text

        score = score.data.cpu().numpy()[0].astype(np.float32)
        text = text.data.cpu().numpy()[0].astype(np.uint8)
        kernels = kernels.data.cpu().numpy()[0].astype(np.uint8)

        pred = pypse(kernels, args.min_kernel_area / (args.scale * args.scale))

        # scale = (org_img.shape[0] * 1.0 / pred.shape[0], org_img.shape[1] * 1.0 / pred.shape[1])
        scale = (org_img.shape[1] * 1.0 / pred.shape[1], org_img.shape[0] * 1.0 / pred.shape[0])
        label = pred
        label_num = np.max(label) + 1
        bboxes = []
        for i in range(1, label_num):
            points = np.array(np.where(label == i)).transpose((1, 0))[:, ::-1]

            if points.shape[0] < args.min_area / (args.scale * args.scale):
                continue

            score_i = np.mean(score[label == i])
            if score_i < args.min_score:
                continue

            rect = cv2.minAreaRect(points)
            bbox = cv2.boxPoints(rect) * scale
            bbox = bbox.astype('int32')
            bboxes.append(bbox.reshape(-1))

        torch.cuda.synchronize()
        end = time.time()
        total_frame += 1
        total_time += (end - start)
        print('fps: %.2f' % (total_frame / total_time))
        sys.stdout.flush()

        for bbox in bboxes:
            cv2.drawContours(text_box, [bbox.reshape(4, 2)], -1, (0, 255, 0), 2)

        image_name = data_loader.img_paths[idx].split('/')[-1].split('.')[0]
        # save contours image
        cv2.imwrite('{}/{}_contours.png'.format(args.outputs_path, image_name), text_box)

        get_coords_and_write(image_name, bboxes, args.outputs_path)

        # text_box = cv2.resize(text_box, (text.shape[1], text.shape[0]))
        # debug(idx, data_loader.img_paths, [[text_box]], args.outputs_path)

    # cmd = 'cd %s;zip -j %s %s/*'%('./outputs/', 'submit_ic15.zip', 'submit_ic15');
    # print(cmd)
    sys.stdout.flush()
    # util.cmd.cmd(cmd)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--CUDA', nargs='?', type=str, default='0')
    parser.add_argument('--scale', nargs='?', type=int, default=1)
    parser.add_argument('--validate_data_path', nargs='?', type=str, default='./data/validate_images/')
    parser.add_argument('--arch', nargs='?', type=str, default='resnet50')
    parser.add_argument('--outputs_path', nargs='?', type=str, default='./outputs/result/')
    parser.add_argument('--model_path', nargs='?', type=str, default='./models/ctw1500_res50_pretrain_ic17.pth.tar')

    parser.add_argument('--binary_th', nargs='?', type=float, default=1.0,
                        help='Path to previous saved model to restart from')
    parser.add_argument('--kernel_num', nargs='?', type=int, default=7,
                        help='Path to previous saved model to restart from')
    parser.add_argument('--long_size', nargs='?', type=int, default=2240,
                        help='Path to previous saved model to restart from')
    parser.add_argument('--min_kernel_area', nargs='?', type=float, default=5.0,
                        help='min kernel area')
    parser.add_argument('--min_area', nargs='?', type=float, default=800.0,
                        help='min area')
    parser.add_argument('--min_score', nargs='?', type=float, default=0.93,
                        help='min score')

    args = parser.parse_args()
    start = time.time()
    test(args)
    end = time.time()
    print('it took {} s'.format(str(int(end - start))))
