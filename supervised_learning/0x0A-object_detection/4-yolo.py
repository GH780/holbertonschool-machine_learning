#!/usr/bin/env python3
"""
Module to perform object detection
"""
import tensorflow.keras as K
import numpy as np
import cv2
import glob


class Yolo:
    """
    uses the Yolo v3 algorithm to perform object detection:
    """

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """
        :param self: instance of the class
        """
        self.model = K.models.load_model(filepath=model_path)
        with open(classes_path, 'r') as f:
            txt_saved = f.read()
            txt_saved = txt_saved.split('\n')
            if len(txt_saved[-1]) == 0:
                txt_saved = txt_saved[:-1]
        self.class_names = txt_saved
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    def process_outputs(self, outputs, image_size):
        """
        :param outputs: is a list of numpy.ndarrays containing the
        predictions from the Darknet model for a single image
        """
        boxes = []
        box_confidence = []
        box_class_probs = []
        img_h, img_w = image_size
        for index, out in enumerate(outputs):
            grid_height, grid_width, anchor_boxes, _ = out.shape
            # Boxes inside
            box = np.zeros(out[:, :, :, :4].shape)
            # Center coordinates, width and height of the output
            t_x = out[:, :, :, 0]
            t_y = out[:, :, :, 1]
            t_w = out[:, :, :, 2]
            t_h = out[:, :, :, 3]

            # Width and height of the predefined anchor
            pw_total = self.anchors[:, :, 0]
            pw = np.tile(pw_total[index], grid_width)
            pw = pw.reshape(grid_width, 1, len(pw_total[index]))
            ph_total = self.anchors[:, :, 1]
            ph = np.tile(ph_total[index], grid_height)
            ph = ph.reshape(grid_height, 1, len(ph_total[index]))

            # Corners of the grid
            cx = np.tile(np.arange(grid_width), grid_height)
            cx = cx.reshape(grid_width, grid_width, 1)
            cy = np.tile(np.arange(grid_width), grid_height)
            cy = cy.reshape(grid_height, grid_height).T
            cy = cy.reshape(grid_height, grid_height, 1)

            # Boxes predictions
            bx = (1 / (1 + np.exp(-t_x))) + cx
            by = (1 / (1 + np.exp(-t_y))) + cy
            bw = np.exp(t_w) * pw
            bh = np.exp(t_h) * ph

            # Normalizing
            bx = bx / grid_width
            by = by / grid_height
            bw = bw / self.model.input.shape[1].value
            bh = bh / self.model.input.shape[2].value

            # Coordinates
            # Top left (x1, y1)
            # Bottom right (x2, y2)
            x1 = (bx - (bw / 2)) * image_size[1]
            y1 = (by - (bh / 2)) * image_size[0]
            x2 = (bx + (bw / 2)) * image_size[1]
            y2 = (by + (bh / 2)) * image_size[0]

            # Append boxes
            box[:, :, :, 0] = x1
            box[:, :, :, 1] = y1
            box[:, :, :, 2] = x2
            box[:, :, :, 3] = y2
            boxes.append(box)

            # Box confidence
            aux = out[:, :, :, 4]
            conf = (1 / (1 + np.exp(-aux)))
            conf = conf.reshape(grid_height, grid_width, anchor_boxes, 1)
            box_confidence.append(conf)

            # Box class probabilities
            aux = out[:, :, :, 5:]
            probs = (1 / (1 + np.exp(-aux)))
            box_class_probs.append(probs)

        return (boxes, box_confidence, box_class_probs)

    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        """
        filters the boxes
        """
        box = [ele.reshape(-1, 4) for ele in boxes]
        box = np.concatenate(box)
        box_scores = []
        for confidence, probs in zip(box_confidences, box_class_probs):
            box_scores.append(confidence * probs)

        classes = [np.argmax(ele, -1) for ele in box_scores]
        classes = [ele.reshape(-1) for ele in classes]
        classes = np.concatenate(classes)

        classes_scores = [np.max(ele, -1) for ele in box_scores]
        classes_scores = [ele.reshape(-1) for ele in classes_scores]
        classes_scores = np.concatenate(classes_scores)

        # mask
        mask = np.where(classes_scores >= self.class_t)

        filtered_boxes = box[mask]
        box_classes = classes[mask]
        box_scores = classes_scores[mask]

        return (filtered_boxes, box_classes, box_scores)

    def nms(self, filtered, thresh, scores):
        """
        function that performs non-maximun suppresion
        """
        x1 = filtered[:, 0]
        y1 = filtered[:, 1]
        x2 = filtered[:, 2]
        y2 = filtered[:, 3]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)

            inds = np.where(ovr <= thresh)[0]
            order = order[inds + 1]

        return keep

    def non_max_suppression(self, filtered_boxes, box_classes, box_scores):
        """
        :param filtered_boxes: a numpy.ndarray of shape (?, 4) containing all
        of the filtered bounding boxes
        """
        unique_classes = np.unique(box_classes)

        all_filtered = []
        all_classes = []
        all_scores = []
        for c in unique_classes:
            idx = np.where(c == box_classes)
            filtered = filtered_boxes[idx]
            classes = box_classes[idx]
            scores = box_scores[idx]

            keep = self.nms(filtered, self.nms_t, scores)

            filtered = filtered[keep]
            classes = classes[keep]
            scores = scores[keep]

            all_filtered.append(filtered)
            all_classes.append(classes)
            all_scores.append(scores)

        # use np.concatenate return np.arrays not lists
        all_filtered = np.concatenate(all_filtered)
        all_classes = np.concatenate(all_classes)
        all_scores = np.concatenate(all_scores)

        return (all_filtered, all_classes, all_scores)

    @staticmethod
    def load_images(folder_path):
        """
        loads images with open cv
        """
        image_paths = glob.glob(folder_path + "/*", recursive=False)
        images = [cv2.imread(img) for img in image_paths]
        return (images, image_paths)
