import xml.etree.ElementTree as ET
import os
import shutil
import argparse


def arg_parse():
    parser = argparse.ArgumentParser(
        description="Get the bndbox area and length-width ratio.")
    parser.add_argument(
        '-x',
        '--xmldir',
        help="files folder of xml ",
        required=True,
    )
    parser.add_argument(
        '-o', '--outfile', required=True, help="filename of result")
    args = parser.parse_args()
    return args


def getboundlength(filename):
    if os.path.splitext(filename)[1] != '.xml':
        return False
    tree = ET.ElementTree(file=filename)
    result = []
    for elem in tree.iterfind('object/bndbox'):
        xmin = int(elem.find('xmin').text)
        ymin = int(elem.find('ymin').text)
        ymax = int(elem.find('ymax').text)
        xmax = int(elem.find('xmax').text)
        result.append([xmax - xmin, ymax - ymin])

    return result


def filterfile(dirs, subdirs, recycledir):
    files = os.listdir(dirs)
    subfiles = os.listdir(subdirs)
    num = 0
    for curfile in subfiles:
        if curfile in files:
            cur_fullpath = os.path.join(dirs, curfile)
            shutil.move(cur_fullpath, recycledir)
            num += 1
    print(num)


def movefile():
    dirs = r'D:\data\数据\VOC2007\JPEGImages'
    subdirs = r'D:\data\数据\小样本'
    deletedir = r'D:\data\数据\delete'
    filterfile(dirs, subdirs, deletedir)


def main():
    # xmldir = r'D:\data\数据\VOC2007\Annotations'
    # resfn = open(r'D:\Code\HTHT\python\loadlabel\小样本.txt', 'w')
    args = arg_parse()
    xmldir = args.xmldir
    outputfile = args.outfile
    if not os.path.isdir(xmldir):
        print('Input xml folder is wrong! %s' % (xmldir))
        return None
    if os.path.exists(outputfile):
        respones = input(
            "The output file is exist: %s\nThis operate will overwrite it?(Y/N)"
        )
        if not (respones == 'Y' or respones == 'y'):
            print("Cancel")
            return None
    resfn = open(outputfile, 'w')
    xmlfiles = os.listdir(xmldir)
    for xmlfile in xmlfiles:
        infos = getboundlength(os.path.join(xmldir, xmlfile))
        if not infos:
            continue
        for info in infos:
            if info[0] and info[1]:
                resfn.write("%d,%d,%d,%.4f\n" % (info[0], 
                    info[1], info[0] * info[1], info[0] / info[1]))
    resfn.close()
    print('Finish!')


if __name__ == '__main__':
    main()