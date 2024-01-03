
import functools
from copy import deepcopy
import xml.etree.ElementTree as ET

import cv2
import numpy as np

class SegmentationTreeCCs:
    def __init__(self, im=None, connectivity=None):
        if im is not None:
            cc_stats = cv2.connectedComponentsWithStats(im, connectivity, cv2.CV_32S)
        # opencv includes the whole image as the first cc for some reason
            self.num_ccs = cc_stats[0] - 1
            self.bboxes = cc_stats[2][1:, :]

    def __len__(self):
        return self.num_ccs

    def filter_by_interval(self, x_limits, y_limits):
        xmin, xmax = x_limits
        ymin, ymax = y_limits
        if self.num_ccs == 0:
            return
        x1, y1, w, h = [self.bboxes[:, i] for i in range(4)]
        x2 = x1 + w
        y2 = y1 + h
        i1 = np.where(x1 >= xmin)[0]
        i2 = np.where(y1 >= ymin)[0]
        i3 = np.where(x2 <= xmax)[0]
        i4 = np.where(y2 <= ymax)[0]
        i = functools.reduce(np.intersect1d, (i1, i2, i3, i4))
        self.num_ccs = len(i)
        self.bboxes = self.bboxes[i, :]

    def get_enclosing_bbox(self, h, w, margin=0):
        bboxes = self.bboxes
        if len(bboxes) == 0:
            return None
        x1 = bboxes[:, 0]
        y1 = bboxes[:, 1]
        x2 = bboxes[:, 0] + bboxes[:, 2]
        y2 = bboxes[:, 1] + bboxes[:, 3]
        xl = max([x1.min() - margin, 0])
        yt = max([y1.min() - margin, 0])
        xr = min([x2.max() + margin, w])
        yb = min([y2.max() + margin, h])

        return (xl, yt, xr - xl, yb - yt)

    def to_xml(self, node_subelement=None):
        node_subelement = ET.Element('CCs') if node_subelement is None else node_subelement
        cc_subelements = ['x', 'y', 'w', 'h', 'a']
        for bbox in self.bboxes:
            cc = ET.SubElement(node_subelement, 'CC')
            for i, subelement in enumerate(cc_subelements):
                ET.SubElement(cc, subelement).text = str(bbox[i])
        return node_subelement

    @staticmethod
    def from_xml(subelement):
        bboxes = []
        cc_subelements = ['x', 'y', 'w', 'h', 'a']
        for cc in subelement.iter('CC'):
            bbox = []
            for subelement in cc_subelements:
                bbox += [int(cc.find(subelement).text)]
            bboxes += [bbox]
        bboxes = np.asarray(bboxes, dtype='int')
        ccs = SegmentationTreeCCs(None, None)
        ccs.num_ccs = len(bboxes)
        ccs.bboxes = bboxes
        return ccs


class SegmentationTreeNode:
    def __init__(self, im, ccs, x_limits, y_limits, H, W):
        self.im = im
        self.ccs = ccs
        self.x_limits = x_limits
        self.y_limits = y_limits
        self.H = H
        self.W = W

        self.left = None
        self.right = None
        self.parent = None
        self.is_leaf = True

        # compute local intervals
        self.compute_local_intervals()
        # filter local ccs by intervals
        self.ccs.filter_by_interval(self.x_limits, self.y_limits)

    def __deepcopy__(self, memodict={}):
        cls = self.__class__
        result = cls.__new__(cls)
        # shallow copy everything first
        result.__dict__.update(self.__dict__)
        # deep copy everything other than im
        memodict[id(self)] = result
        for k, v in self.__dict__.items():
            if k != 'im':
                setattr(result, k, deepcopy(v, memodict))

        return result

    def compute_local_intervals(self):
        if self.im is None:
            return
        # print(self.y_limits)
        # print(self.x_limits)
        local_im = self.im[self.y_limits[0] : self.y_limits[1], self.x_limits[0] : self.x_limits[1]]
        hpp1 = SegmentationTreeNode.horizontal_pixel_profile(local_im)
        vpp1 = SegmentationTreeNode.vertical_pixel_profile(local_im)
        self.x_intervals = SegmentationTreeNode.get_all_cut_intervals(vpp1)
        self.y_intervals = SegmentationTreeNode.get_all_cut_intervals(hpp1)
        self.x_intervals += self.x_limits[0]
        self.y_intervals += self.y_limits[0]
        # print('cut intervals for', self.x_limits, self.y_limits, self.im.shape, local_im.shape)
        # print(self.x_intervals)
        # print(self.y_intervals)
        # if 0 not in self.im.shape:
        #     cv2.imwrite('__SEGMENTATION/{}_{}_{}_{}_img.jpg'.format(self.x_limits[0], self.x_limits[1], self.y_limits[0], self.y_limits[1]), local_im)
        # else:
        #     cv2.imwrite('SEGMENTATION_{}_{}_{}_{}_img.jpg'.format(self.x_limits[0], self.x_limits[1], self.y_limits[0],
        #                                                           self.y_limits[1]), np.zeros(shape=(1, 1)))

    def segment(self, alpha_x, alpha_y):
        # print("->here")
        # print(self.ccs)
        # check if valid segmentation exists if not return
        # print('at node', self.x_limits, self.y_limits)
        if len(self.ccs) == 0:
            # print('no CCs. no segmentation', self.left, self.right)
            # print("->> |ccs| = 0")
            return
        if len(self.x_intervals) == 0 and len(self.y_intervals) == 0:
            # print('no cut intervals. no segmentation', self.left, self.right)
            # print("->> No intervals")
            return
        # compute local thresholds
        xthr, ythr = SegmentationTreeNode.get_xy_cut_thresholds(self.ccs.bboxes, alpha_x, alpha_y)
        # check if any x segmentation is possible
        if len(self.x_intervals) > 0:
            widths = self.x_intervals[:, 1] - self.x_intervals[:, 0]
            max_cut_width = widths.max()
            max_cut_width = max_cut_width if max_cut_width >= xthr else 0
            best_x_cut = self.x_intervals[np.argmax(widths), :]
        else:
            max_cut_width = 0
        # check if any y segmentation is possible
        if len(self.y_intervals) > 0:
            heights = self.y_intervals[:, 1] - self.y_intervals[:, 0]
            max_cut_height = heights.max()
            max_cut_height = max_cut_height if max_cut_height >= ythr else 0
            best_y_cut = self.y_intervals[np.argmax(heights), :]
        else:
            max_cut_height = 0
        # if the best X and Y cuts do not cross the minimum computed thresholds
        if max_cut_height == 0 and max_cut_width == 0:
            # print('no valid x or y cut intervals greater than threshold')
            # print("->> max cut w/h")
            return
        # if we have survived till this point some segmentation is possible
        self.is_leaf = False
        # pick the best possible segmentation based on number of pixels
        if max_cut_height >= max_cut_width:
            y1, y2 = best_y_cut
            # print('Y segmenting at', y1, y2)
            self.left = SegmentationTreeNode(self.im, deepcopy(self.ccs),
                                             self.x_limits, (self.y_limits[0], y1), self.H, self.W)

            self.right = SegmentationTreeNode(self.im, deepcopy(self.ccs),
                                              self.x_limits, (y2, self.y_limits[1]), self.H, self.W)
        else:
            x1, x2 = best_x_cut
            # print('X segmenting at', x1, x2)
            self.left = SegmentationTreeNode(self.im, deepcopy(self.ccs),
                                             (self.x_limits[0], x1), self.y_limits, self.H, self.W)

            self.right = SegmentationTreeNode(self.im, deepcopy(self.ccs),
                                              (x2, self.x_limits[1]), self.y_limits, self.H, self.W)
        self.left.parent = self
        self.right.parent = self

    def force_segment_Y(self, y):
        self.is_leaf = False
        y = int(y)
        # print('Y segmenting at', y1, y2)
        self.left = SegmentationTreeNode(self.im, deepcopy(self.ccs),
                                         self.x_limits, (self.y_limits[0], y), self.H, self.W)

        self.right = SegmentationTreeNode(self.im, deepcopy(self.ccs),
                                          self.x_limits, (y + 1, self.y_limits[1]), self.H, self.W)
        self.left.parent = self
        self.right.parent = self

    def force_segment_X(self, x):
        self.is_leaf = False
        x = int(x)
        self.left = SegmentationTreeNode(self.im, deepcopy(self.ccs),
                                         (self.x_limits[0], x), self.y_limits, self.H, self.W)

        self.right = SegmentationTreeNode(self.im, deepcopy(self.ccs),
                                          (x + 1, self.x_limits[1]), self.y_limits, self.H, self.W)
        self.left.parent = self
        self.right.parent = self

    def to_xml(self, node_subelement=None):
        node_subelement = ET.Element('root') if node_subelement is None else node_subelement
        node_ccs = ET.SubElement(node_subelement, 'CCs')
        self.ccs.to_xml(node_ccs)
        x_limits = ET.SubElement(node_subelement, 'X_Limits')
        ET.SubElement(x_limits, 'x1').text = str(self.x_limits[0])
        ET.SubElement(x_limits, 'x2').text = str(self.x_limits[1])
        y_limits = ET.SubElement(node_subelement, 'Y_Limits')
        ET.SubElement(y_limits, 'y1').text = str(self.y_limits[0])
        ET.SubElement(y_limits, 'y2').text = str(self.y_limits[1])
        ET.SubElement(node_subelement, 'H').text = str(self.H)
        ET.SubElement(node_subelement, 'W').text = str(self.W)
        ET.SubElement(node_subelement, 'is_leaf').text = str(self.is_leaf)

        if self.left is not None:
            left = ET.SubElement(node_subelement, 'left')
            self.left.to_xml(left)
        if self.right is not None:
            right = ET.SubElement(node_subelement, 'right')
            self.right.to_xml(right)
        return node_subelement

    @staticmethod
    def from_xml(node_subelement, bin_image):
        ccs_subelement = node_subelement.find('CCs')
        if ccs_subelement is None:
            ccs = SegmentationTreeCCs(None, None)
            ccs.num_ccs = 0
            ccs.bboxes = np.empty(shape=(0, 4))
        ccs = SegmentationTreeCCs.from_xml(ccs_subelement)
        xl = node_subelement.find('X_Limits')
        x_limits = (int(xl.find('x1').text), int(xl.find('x2').text))
        yl = node_subelement.find('Y_Limits')
        y_limits = (int(yl.find('y1').text), int(yl.find('y2').text))
        H = int(node_subelement.find('H').text)
        W = int(node_subelement.find('W').text)
        node = SegmentationTreeNode(bin_image, ccs, x_limits, y_limits, H, W)
        node.is_leaf = node_subelement.find('is_leaf').text == 'True'
        # print(node, is_leaf, node.x_limits, node.y_limits, len(node.ccs), node.ccs.bboxes.shape)
        if not node.is_leaf:
            left_subelement = node_subelement.find('left')
            node.left = SegmentationTreeNode.from_xml(left_subelement, bin_image)
            node.left.parent = node
            right_subelement = node_subelement.find('right')
            node.right = SegmentationTreeNode.from_xml(right_subelement, bin_image)
            node.right.parent = node
        return node

    @staticmethod
    def horizontal_pixel_profile(im):
        return im.astype(np.float32).sum(axis=1)

    @staticmethod
    def vertical_pixel_profile(im):
        return im.astype(np.float32).sum(axis=0)

    @staticmethod
    def zero_runs(a):
        # Create an array that is 1 where a is 0, and pad each end with an extra 0.
        iszero = np.concatenate(([0], np.equal(a, 0).view(np.int8), [0]))
        absdiff = np.abs(np.diff(iszero))
        # Runs start and end where absdiff is 1.
        ranges = np.where(absdiff == 1)[0].reshape(-1, 2)
        return ranges

    @staticmethod
    def get_all_cut_intervals(profile):
        gaps = SegmentationTreeNode.zero_runs(profile)
        return gaps

    @staticmethod
    def get_xy_cut_thresholds(bboxes, alpha_x, alpha_y):
        # def get_xy_cut_thresholds(bboxes, alpha_x=-1.25, alpha_y=-3):
        widths = bboxes[:, 2]
        heights = bboxes[:, 3]
        wmean = np.mean(widths)
        wstd = np.std(widths)
        hmean = np.mean(heights)
        hstd = np.std(heights)
        xthr = max([wmean + (alpha_x * wstd), 3])
        ythr = max([hmean + (alpha_y * hstd), 3])

        # print(wmean, hmean, wstd, hstd, xthr, ythr)
        return xthr, ythr

class SegmentationTree:
    def __init__(self, root_node):
        self.root = root_node
        self.root.parent = self.root
        self.visited = set()
        self.visited.add(self.root)

    # a template to traverse the tree
    def __traverse_tree(self, node):
        self.visited.add(node)
        # do node action
        while node.left is not None and node.left not in self.visited:
            # call recursively at left node
            self.__traverse_tree(node.left)
        node = node.parent
        while node.right is not None and node.right not in self.visited:
            # call recursively at right node
            self.__traverse_tree(node.right)

    def segment(self, node, alpha_x, alpha_y):
        self.visited.add(node)
        node.segment(alpha_x, alpha_y)
        while node.left is not None and node.left not in self.visited:
            self.segment(node.left, alpha_x, alpha_y)
        node = node.parent
        while node.right is not None and node.right not in self.visited:
            self.segment(node.right, alpha_x, alpha_y)

    def find_bbox_by_coords(self, x, y, node, tight=False):
        if node is None:
            # print("None 1")
            return None, None
        x1, x2 = node.x_limits
        y1, y2 = node.y_limits
        # print(node.x_limits, node.y_limits)
        if x1 <= x <= x2 and y1 <= y <= y2:
            # print('valid query at', node.x_limits, node.y_limits)
            left = node.left
            right = node.right
            # if valid segmentation exists at this node we need to go further to find a leaf
            if left is not None and right is not None:
                left_x1, left_x2 = node.left.x_limits
                right_x1, right_x2 = node.right.x_limits
                left_y1, left_y2 = node.left.y_limits
                right_y1, right_y2 = node.right.y_limits
                # go to left sub-tree to recursively continue search
                if left_x1 <= x <= left_x2 and left_y1 <= y <= left_y2:
                    return self.find_bbox_by_coords(x, y, left)
                # go to right sub-tree to recursively continue search
                elif right_x1 <= x <= right_x2 and right_y1 <= y <= right_y2:
                    return self.find_bbox_by_coords(x, y, right)
                # a segmentation exists but the given set of coords does not fall into either segmentation so no box
                else:
                    # print("None 2")
                    # print(((x, y), ((left_x1, left_x2), (left_y1, left_y2)), ((right_x1, right_x2), (right_y1, right_y2)) ))
                    return None, None
            # we are at a leaf node and the coords are within the leafs interval so return the enclosing box of the leaf
            else:
                # print(len(node.ccs), node.ccs.bboxes, node.ccs.get_enclosing_bbox(node.H, node.W))
                bbox = node.ccs.get_enclosing_bbox(node.H, node.W, 3)
                if not tight:
                    return bbox, node
                # if tight flag is True check if the query point inside the boundaries of all CCs in the node
                if bbox[0] <= x <= bbox[0] + bbox[2] and bbox[1] <= y <= bbox[1] + bbox[3]:
                    return bbox, node
                else:
                    # print("None 3")
                    return None, None
        # we are at a leaf and the coords dont exist in the interval (to deal with out of range coords)
        else:
            # print("None 4")
            return None, None

    # correct oversegmentation
    def remove_segment(self, node):
        parent = node.parent

        to_remove = [parent.left, parent.right]
        pos_remove = 0
        while pos_remove < len(to_remove):
            next_remove = to_remove[pos_remove]
            # if the node is not a leaf ... remove children
            if not next_remove.is_leaf:
                to_remove.append(next_remove.left)
                to_remove.append(next_remove.right)
            self.visited.remove(next_remove)
            pos_remove += 1
        # self.visited.remove(parent.left)
        # self.visited.remove(parent.right)

        parent.left = None
        parent.right = None
        parent.is_leaf = True

    # horizontal split
    def force_segment_Y(self, y, node):
        node.force_segment_Y(y)
        self.__traverse_tree(node)

    # vertical split
    def force_segment_X(self, x, node):
        node.force_segment_X(x)
        self.__traverse_tree(node)

    def collect_all_leaves(self):
        bboxes = []
        if len(self.visited) == 0:
            print('no root in this tree!')
        leaf_nodes = [node for node in self.visited if node.is_leaf]
        for node in leaf_nodes:
            bbox = node.ccs.get_enclosing_bbox(node.H, node.W, 3)
            if bbox is not None:
                bboxes += [bbox]
        return bboxes

    def to_xml(self):
        xml_tree_root = ET.Element('SegmentationTree')
        self.root.to_xml(xml_tree_root)

        return ET.tostring(xml_tree_root).decode('utf-8') + "\n"


    def update_image(self, bin_image):
        if len(bin_image.shape) == 3:
            bin_image = bin_image[:, :, 0]
        for node in self.visited:
            node.im = bin_image
            node.compute_local_intervals()

    @staticmethod
    def from_xml(root_xml, bin_image):
        # tree_xml = ET.parse(filename)
        # root_xml = tree_xml.getroot()
        root = SegmentationTreeNode.from_xml(root_xml, bin_image)
        tree = SegmentationTree(root)
        tree.__traverse_tree(tree.root)
        return tree

    @staticmethod
    def SegmentationTreesToXML(tree_array):
        xml_str = "   <VideoKeyFramesWords>\n"
        for tree in tree_array:
            xml_str += tree.to_xml()
        xml_str += "   </VideoKeyFramesWords>\n"

        return xml_str

    @staticmethod
    def LoadSegmentationTreesFromXML(xml_filename, namespace, bin_images):
        tree = ET.parse(xml_filename)
        root = tree.getroot()

        trees_root = root.find(namespace + "VideoKeyFramesWords")
        loaded_trees = []
        for idx, kf_tree_root in enumerate(trees_root):
            kf_tree = SegmentationTree.from_xml(kf_tree_root, bin_images[idx])
            loaded_trees.append(kf_tree)

        return loaded_trees

    @staticmethod
    def CreateDefault(bin_image):
        if len(bin_image.shape) == 3:
            bin_image = bin_image[:, :, 0]

        h, w = bin_image.shape

        ccs = SegmentationTreeCCs(bin_image, 8)
        root = SegmentationTreeNode(bin_image, ccs, (0, w), (0, h), h, w)

        seg = SegmentationTree(root)

        return seg
