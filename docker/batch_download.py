import os
import platform
import pip
# pip V10.0.0以上版本需要导入下面的包
from pip._internal.utils.misc import get_installed_distributions
import pdb


def download_package(packagefolder):
    print('current platform system is %s' % (platform.system()))
    wrongfn = open("./log_download_package.txt", "w")
    with open(packagefolder) as fn:
        packageinfo = fn.readline()[:-1]
        while packageinfo:
            if os.system(r'pip3 download ' + packageinfo):
                print("pip3 download %s sucessful" % (packageinfo))
            else:
                wrongfn.write(packageinfo + "\n")
            packageinfo = fn.readline()[:-1]


def install_online_package(packagefolder):
    print('current platform system is %s' % (platform.system()))
    wrongfn = open("./log_install_online_package.txt", "w")
    with open(packagefolder) as fn:
        packageinfo = fn.readline()[:-1]
        while packageinfo:
            if os.system(r'pip install ' + packageinfo):
                print("pip install %s sucessful" % (packageinfo))
            else:
                wrongfn.write(packageinfo + "\n")
            packageinfo = fn.readline()[:-1]


def install_offline_package(packagefolder):
    print('current platform system is %s' % (platform.system()))
    packagelist = os.listdir(packagefolder)
    i = 0
    num = len(packagelist)
    # if str(platform.system()) == 'Windows':
    while len(packagelist):
        if num != len(packagelist):
            num = len(packagelist)
            i = 0
        if i > num:
            break
        if len(packagelist[0]) > 4:
            if packagelist[0][-4:] not in [".whl"]:
                del packagelist[0]
                continue
        else:
            del packagelist[0]
            continue
        i = i + 1
        package_absdir = packagefolder + '/' + packagelist[0]
        if os.path.isfile(package_absdir):
            if os.system(r'pip install ' + package_absdir):
                print(r'pip install %s failed!' % packagelist[0])
                packagelist.append(packagelist[0])
            else:
                print(r'pip install %s successeded!' % packagelist[0])
            del packagelist[0]
            print('len(packagelist): %d' % len(packagelist))


def main():
    # download_package(r"./re.txt")
    # install_offline_package(os.getcwd())
    install_online_package(r"./re.txt")


if __name__ == '__main__':
    main()
