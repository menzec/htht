import argparse
import os
import xml.etree.ElementTree as ET


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


def main():
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
                resfn.write(
                    "%d,%d,%d,%.4f\n" % (info[0], info[1], info[0] * info[1],
                                         info[0] / info[1]))
    resfn.close()
    print('Finish!')


if __name__ == '__main__':
    main()
