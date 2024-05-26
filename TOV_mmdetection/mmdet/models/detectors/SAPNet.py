import copy
from tkinter import N
import mmcv
from ..builder import DETECTORS
from .two_stage import TwoStageDetector
from mmdet.core.bbox import bbox_xyxy_to_cxcywh
from mmdet.core.bbox import bbox_cxcywh_to_xyxy
import torch
import numpy as np
from mmdet.core.visualization.image import imshow_det_bboxes
from mmdet.core.bbox.iou_calculators import bbox_overlaps
from ..builder import build_head


def gen_negative_proposals(gt_points, proposal_cfg, aug_generate_proposals, img_meta, pro_score, stage):
    num_neg_gen = proposal_cfg['gen_num_neg']
    if num_neg_gen == 0:
        return None, None
    neg_proposal_list = []
    neg_weight_list = []
    for i in range(len(gt_points)):
        num_gt = len(gt_points[i])
        proposals = aug_generate_proposals[i].view(num_gt, -1,aug_generate_proposals[0].shape[-1])
        
        pos_bboxes = aug_generate_proposals[i]
        h, w, _ = img_meta[i]['img_shape']
        x1 = -0.2 * w + torch.rand(num_neg_gen) * (1.2 * w)
        y1 = -0.2 * h + torch.rand(num_neg_gen) * (1.2 * h)
        x2 = x1 + torch.rand(num_neg_gen) * (1.2 * w - x1)
        y2 = y1 + torch.rand(num_neg_gen) * (1.2 * h - y1)
        neg_bboxes = torch.stack([x1, y1, x2, y2], dim=1).to(gt_points[0].device)
        #gt_point = gt_points[i]
        # gt_min_box = torch.cat([gt_point - 10, gt_point + 10], dim=1)
        iou = bbox_overlaps(neg_bboxes, pos_bboxes)
        neg_weight = ((iou < 0.3).sum(dim=1) == iou.shape[1])
        # for j in range(num_gt):
        #     pos_box= proposals[j][torch.argmax(pro_score[j])]# add by wzy
        #     pos_box = pos_box.view(1, pos_box.shape[-1])# add by wzy
        #     neg_bbox =  proposals[j][torch.argmin(pro_score[j])]#add by wzy
        #     neg_bbox = neg_bbox.view(1, neg_bbox.shape[-1])# add by wzy
        #     iou1 = bbox_overlaps(neg_bbox, pos_box)
        #     if iou1<0.1:
        #         neg_bboxes = torch.cat((neg_bboxes, neg_bbox),0)
        #         neg_weight1 = ((iou1 < 0.1).sum(dim=1) == iou1.shape[1])
        #         neg_weight = torch.cat((neg_weight, neg_weight1),0)
        #     else:
        #         continue
        if stage ==1:
            for j in range(num_gt):
                pos_box= proposals[j][torch.argmax(pro_score[i][j])]# add by wzy
                pos_box = pos_box.view(1, pos_box.shape[-1])# add by wzy
                w_box = pos_box[0][2]-pos_box[0][0]
                h_box =  pos_box[0][3]-pos_box[0][1]
                if w_box*h_box > 64*64:
                    center_x_negbbox =  (pos_box[0][2]+pos_box[0][0])/2
                    center_y_negbbox =  (pos_box[0][3]+pos_box[0][1])/2
                    a = torch.unsqueeze(center_x_negbbox-(w_box/4), dim=0)
                    b = torch.unsqueeze(center_y_negbbox-(h_box/4), dim=0)
                    c = torch.unsqueeze(center_x_negbbox+(w_box/4), dim=0)
                    d = torch.unsqueeze(center_y_negbbox+(h_box/4), dim=0)  #dui duoge tensor yaoxian unsqueeze caineng cat
                    neg_bbox = torch.cat((a, b, c, d), 0)
                else:
                    continue 
                #neg_bbox =  proposals[j][torch.argmin(pro_score[j])]#add by wzy
                neg_bbox = neg_bbox.view(1, neg_bbox.shape[-1])# add by wzy
                iou1 = bbox_overlaps(neg_bbox, pos_box)
                if iou1<0.5:
                    neg_bboxes = torch.cat((neg_bboxes, neg_bbox),0)
                    neg_weight1 = ((iou1 < 0.5).sum(dim=1) == iou1.shape[1])
                    neg_weight = torch.cat((neg_weight, neg_weight1),0)
                else:
                    continue
        neg_proposal_list.append(neg_bboxes)
        neg_weight_list.append(neg_weight)
        #select neg from proposals
        
    return neg_proposal_list, neg_weight_list


@DETECTORS.register_module()
class SAPNet(TwoStageDetector):
    def __init__(self,
                 backbone,
                 roi_head,
                 train_cfg,
                 test_cfg,
                 bbox_head=None,
                 mask_head=None,  
                #  affinity_head=None, #gai by wzy
                 neck=None,
                 pretrained=None,
                 init_cfg=None):
        super(SAPNet, self).__init__(
            backbone=backbone,
            neck=neck,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
            init_cfg=init_cfg)
        self.num_stages = roi_head.num_stages
        if bbox_head is not None:
            self.with_bbox_head = True
            self.bbox_head = build_head(bbox_head)
        assert mask_head, f'`mask_head` must ' \
                          f'be implemented in {self.__class__.__name__}'
        mask_head.update(train_cfg=copy.deepcopy(train_cfg))
        mask_head.update(test_cfg=copy.deepcopy(test_cfg))
        self.mask_head = build_head(mask_head)  #gai by wzy

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_masks ,  #gai by wzy
                      gt_true_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      **kwargs):
        x = self.extract_feat(img)
        gt_masks = [
            gt_mask.to_tensor(dtype=torch.bool, device=img.device)
            for gt_mask in gt_masks
        ]
        base_proposal_cfg = self.train_cfg.get('base_proposal',
                                                self.test_cfg.rpn)
        fine_proposal_cfg = self.train_cfg.get('fine_proposal',
                                               self.test_cfg.rpn)
        epoch=None
        losses = dict()
        gt_points = [bbox_xyxy_to_cxcywh(b)[:, :2] for b in gt_true_bboxes]
        for stage in range(self.num_stages):
            if stage == 0:
                generate_proposals=gt_bboxes
                proposals_valid_list = []
                pro_scores = []
                for m in range(len(gt_bboxes)):
                    for i in range(len(gt_bboxes[m])):
                        base_proposals = []
                        base_proposals =  gt_bboxes[m][i]
                        proposals_valid = base_proposals.new_full(
                        (*base_proposals.shape[:-1], 1), 1, dtype=torch.long).reshape(-1, 1)
                        proposals_valid_list.append(proposals_valid)
                dynamic_weight = torch.cat(gt_labels).new_ones(len(torch.cat(gt_labels)))
                neg_proposal_list, neg_weight_list = None, None
                pseudo_boxes = generate_proposals
            elif stage == 1:
                generate_proposals_pre = gt_bboxes
                generate_proposals = proposals_pro 
                proposals_valid_list = []
                for m in range(len(generate_proposals)):
                    for i in range(len(generate_proposals[m])):
                        base_proposals = []
                        base_proposals =  generate_proposals[m][i]
                        proposals_valid = base_proposals.new_full(
                        (*base_proposals.shape[:-1], 1), 1, dtype=torch.long).reshape(-1, 1)
                        proposals_valid_list.append(proposals_valid)
                #neg_proposal_list, neg_weight_list = None, None
                neg_proposal_list, neg_weight_list = gen_negative_proposals(gt_points, fine_proposal_cfg,
                                                                            generate_proposals_pre,
                                                                            img_meta=img_metas, pro_score =pro_scores, stage=stage)

            roi_losses, pseudo_boxes, dynamic_weight, pro_scores, proposals_pro = self.roi_head.forward_train(stage,epoch, x, img_metas,
                                                                                   pseudo_boxes,
                                                                                   generate_proposals,
                                                                                   proposals_valid_list,
                                                                                   neg_proposal_list, neg_weight_list,
                                                                                   gt_true_bboxes, gt_labels,
                                                                                   dynamic_weight,
                                                                                   gt_bboxes_ignore, gt_masks,
                                                                                   **kwargs)
            batch_gt = [len(b) for b in gt_points]
            pro_scores = torch.split(pro_scores, batch_gt)
            #if stage==1:
            if stage==1:
                pseudo_masks  = list()
                mask_list = list()
                for i in range(len(gt_bboxes)):
                    bboxes = gt_bboxes[i].reshape(-1,6,4)   
                    pseudo_box = pseudo_boxes[i].unsqueeze(1).expand(-1, 6, -1)
                    iou_bbox = bbox_overlaps(bboxes[:], pseudo_box[:])
                    iou_sum = iou_bbox.sum(dim=-1)
                    max_indices = torch.argmax(iou_sum, dim=-1)
                    pseudo_boxes[i][range(len(max_indices)) , 2] = pseudo_boxes[i][range(len(max_indices)) , 2] + 1
                    pseudo_boxes[i][range(len(max_indices)) , 3] = pseudo_boxes[i][range(len(max_indices)), 3] + 1
                    gt_mask = gt_masks[i].reshape(-1,6,gt_masks[i].shape[-2], gt_masks[i].shape[-1])
                    #mask paixu
                    sorted_scores, indices = pro_scores[i][:,0:6].sort(dim=1, descending=True)
                    sorted_gt_masks = gt_mask.gather(1, indices.unsqueeze(-1).unsqueeze(-1).expand_as(gt_mask))
                    masks = (gt_mask[range(len(max_indices)), max_indices])
                    mask_sort = sorted_gt_masks.permute(1,0,2,3)  #[6,n,h,w]
                    masks = (gt_mask[range(len(max_indices)), max_indices])
                    pseudo_masks.append(masks)
                    mask_list.append(mask_sort)
                mask_loss = self.mask_head.forward_train(
                    epoch,
                    x,
                    gt_labels,
                    pseudo_masks,
                    img,
                    mask_list,
                    img_metas,
                    positive_infos=None,
                    gt_bboxes=pseudo_boxes,
                    gt_bboxes_ignore=gt_bboxes_ignore,
                    **kwargs)  
            for key, value in roi_losses.items():
                losses[f'stage{stage}_{key}'] = value
            if stage == 0:
                pseudo_boxes_out = pseudo_boxes
                dynamic_weight_out = dynamic_weight
            elif stage == 1:
                losses.update(mask_loss)  #gai by wzy
        return losses

    def simple_test(self, img, img_metas, rescale=False):
            """Test function without test-time augmentation.

            Args:
                img (torch.Tensor): Images with shape (B, C, H, W).
                img_metas (list[dict]): List of image information.
                rescale (bool, optional): Whether to rescale the results.
                    Defaults to False.

            Returns:
                list(tuple): Formatted bbox and mask results of multiple \
                    images. The outer list corresponds to each image. \
                    Each tuple contains two type of results of single image:

                    - bbox_results (list[np.ndarray]): BBox results of
                    single image. The list corresponds to each class.
                    each ndarray has a shape (N, 5), N is the number of
                    bboxes with this category, and last dimension
                    5 arrange as (x1, y1, x2, y2, scores).
                    - mask_results (list[np.ndarray]): Mask results of
                    single image. The list corresponds to each class.
                    each ndarray has a shape (N, img_h, img_w), N
                    is the number of masks with this category.
            """
            feat = self.extract_feat(img)
            results_list = None
            results_list = self.mask_head.simple_test(img, 
                feat, img_metas, rescale=rescale, instances_list=results_list)

            format_results_list = []
            for results in results_list:
                format_results_list.append(self.format_results_SAM(results))

            return format_results_list

    def format_results(self, results):
        """Format the model predictions according to the interface with
        dataset.

        Args:
            results (:obj:`InstanceData`): Processed
                results of single images. Usually contains
                following keys.

                - scores (Tensor): Classification scores, has shape
                    (num_instance,)
                - labels (Tensor): Has shape (num_instances,).
                - masks (Tensor): Processed mask results, has
                    shape (num_instances, h, w).

        Returns:
            tuple: Formatted bbox and mask results.. It contains two items:

                - bbox_results (list[np.ndarray]): BBox results of
                    single image. The list corresponds to each class.
                    each ndarray has a shape (N, 5), N is the number of
                    bboxes with this category, and last dimension
                    5 arrange as (x1, y1, x2, y2, scores).
                - mask_results (list[np.ndarray]): Mask results of
                    single image. The list corresponds to each class.
                    each ndarray has shape (N, img_h, img_w), N
                    is the number of masks with this category.
        """
        data_keys = results.keys()
        assert 'scores' in data_keys
        assert 'labels' in data_keys

        assert 'masks' in data_keys, \
            'results should contain ' \
            'masks when format the results '
        mask_results = [[] for _ in range(self.mask_head.num_classes)]

        num_masks = len(results)

        if num_masks == 0:
            bbox_results = [
                np.zeros((0, 5), dtype=np.float32)
                for _ in range(self.mask_head.num_classes)
            ]
            return bbox_results, mask_results

        labels = results.labels.detach().cpu().numpy()

        if 'bboxes' not in results:
            # create dummy bbox results to store the scores
            results.bboxes = results.scores.new_zeros(len(results), 4)

        det_bboxes = torch.cat([results.bboxes, results.scores[:, None]],
                                dim=-1)
        det_bboxes = det_bboxes.detach().cpu().numpy()
        bbox_results = [
            det_bboxes[labels == i, :]
            for i in range(self.mask_head.num_classes)
        ]

        masks = results.masks.detach().cpu().numpy()

        for idx in range(num_masks):
            mask = masks[idx]
            mask_results[labels[idx]].append(mask)

        return bbox_results, mask_results
    
    def format_results_SAM(self, results):
        bbox_results = [[] for _ in range(self.mask_head.num_classes)]
        mask_results = [[] for _ in range(self.mask_head.num_classes)]
        score_results = [[] for _ in range(self.mask_head.num_classes)]

        for cate_label, cate_score, seg_mask in zip(results.labels, results.scores, results.masks):
            if seg_mask.sum() > 0:
                mask_results[cate_label].append(seg_mask.cpu())
                score_results[cate_label].append(cate_score.cpu())
                ys, xs = torch.where(seg_mask)
                min_x, min_y, max_x, max_y = xs.min().cpu().data.numpy(), ys.min().cpu().data.numpy(), xs.max().cpu().data.numpy(), ys.max().cpu().data.numpy()
                bbox_results[cate_label].append([min_x, min_y, max_x+1, max_y+1, cate_score.cpu().data.numpy()])

        bbox_results = [np.array(bbox_result) if len(bbox_result) > 0 else np.zeros((0, 5)) for bbox_result in bbox_results]

        return bbox_results, (mask_results, score_results)

    def aug_test(self, imgs, img_metas, rescale=False):
        raise NotImplementedError


    def show_result(self,
                    img,
                    result,
                    score_thr=0.3,
                    bbox_color=(72, 101, 241),
                    text_color=(72, 101, 241),
                    mask_color=None,
                    thickness=2,
                    font_size=13,
                    win_name='',
                    show=False,
                    wait_time=0,
                    out_file=None):
        """Draw `result` over `img`.

        Args:
            img (str or Tensor): The image to be displayed.
            result (Tensor or tuple): The results to draw over `img`
                bbox_result or (bbox_result, segm_result).
            score_thr (float, optional): Minimum score of bboxes to be shown.
                Default: 0.3.
            bbox_color (str or tuple(int) or :obj:`Color`):Color of bbox lines.
               The tuple of color should be in BGR order. Default: 'green'
            text_color (str or tuple(int) or :obj:`Color`):Color of texts.
               The tuple of color should be in BGR order. Default: 'green'
            mask_color (None or str or tuple(int) or :obj:`Color`):
               Color of masks. The tuple of color should be in BGR order.
               Default: None
            thickness (int): Thickness of lines. Default: 2
            font_size (int): Font size of texts. Default: 13
            win_name (str): The window name. Default: ''
            wait_time (float): Value of waitKey param.
                Default: 0.
            show (bool): Whether to show the image.
                Default: False.
            out_file (str or None): The filename to write the image.
                Default: None.

        Returns:
            img (Tensor): Only if not `show` or `out_file`
        """
        img = mmcv.imread(img)
        img = img.copy()
        if isinstance(result, tuple):
            bbox_result, segm_result = result
            if isinstance(segm_result, tuple):
                segm_result = segm_result[0]  # ms rcnn
        else:
            bbox_result, segm_result = result, None
        bboxes = np.vstack(bbox_result)
        labels = [
            np.full(bbox.shape[0], i, dtype=np.int32)
            for i, bbox in enumerate(bbox_result)
        ]
        labels = np.concatenate(labels)
        # draw segmentation masks
        segms = None
        if segm_result is not None and len(labels) > 0:  # non empty
            segms = mmcv.concat_list(segm_result)
            if isinstance(segms[0], torch.Tensor):
                segms = torch.stack(segms, dim=0).detach().cpu().numpy()
            else:
                segms = np.stack(segms, axis=0)
        # if out_file specified, do not show image in window
        if out_file is not None:
            show = False
        # draw bounding boxes
        img = imshow_det_bboxes(
            img,
            bboxes,
            labels,
            segms,
            class_names=self.CLASSES,
            score_thr=score_thr,
            bbox_color=bbox_color,
            text_color=text_color,
            mask_color=mask_color,
            thickness=thickness,
            font_size=font_size,
            win_name=win_name,
            show=show,
            wait_time=wait_time,
            out_file=out_file)

        if not (show or out_file):
            return img

