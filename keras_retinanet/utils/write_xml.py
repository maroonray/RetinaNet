from lxml.etree import Element, SubElement, tostring
from xml.dom.minidom import parseString


def make_xml(xmin_tuple, ymin_tuple, xmax_tuple, ymax_tuple, label_tuple, image_name, image_width, image_height):
    node_root = Element('annotation')

    node_folder = SubElement(node_root, 'folder')
    node_folder.text = 'VOC'

    node_filename = SubElement(node_root, 'filename')
    node_filename.text = image_name

    node_object_num = SubElement(node_root, 'path')
    node_object_num.text = './VOC2007/JPEGImages/' + image_name

    node_object_source = SubElement(node_root, 'source')
    node_source_database = SubElement(node_object_source, 'database')
    node_source_database.text = 'Unknown'

    node_object_segmented = SubElement(node_root, 'segmented')
    node_object_segmented.text = '0'

    node_size = SubElement(node_root, 'size')
    node_width = SubElement(node_size, 'width')
    node_width.text = str(int(image_width))

    node_height = SubElement(node_size, 'height')
    node_height.text = str(int(image_height))

    node_depth = SubElement(node_size, 'depth')
    node_depth.text = '3'

    for i in range(len(xmin_tuple)):
        node_object = SubElement(node_root, 'object')
        node_name = SubElement(node_object, 'name')
        node_name.text = label_tuple[i]
        node_difficult = SubElement(node_object, 'difficult')
        node_difficult.text = '0'

        node_pose = SubElement(node_object, 'pose')
        node_pose.text = 'Unspecified'

        node_truncated = SubElement(node_object, 'truncated')
        node_truncated.text = '0'

        node_bndbox = SubElement(node_object, 'bndbox')
        node_xmin = SubElement(node_bndbox, 'xmin')
        node_xmin.text = str(int(xmin_tuple[i]))
        node_ymin = SubElement(node_bndbox, 'ymin')
        node_ymin.text = str(int(ymin_tuple[i]))
        node_xmax = SubElement(node_bndbox, 'xmax')
        node_xmax.text = str(int(xmax_tuple[i]))
        node_ymax = SubElement(node_bndbox, 'ymax')
        node_ymax.text = str(int(ymax_tuple[i]))

    xml = tostring(node_root, pretty_print=True)
    dom = parseString(xml)

    return dom