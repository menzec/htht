import os


def batchinstall(packagedir):
    packagelist = os.listdir(packagedir)
    num = len(packagelist)
    i = 0
    while len(packagelist):
        if num != len(packagelist):
            num = len(packagelist)
            # tem = len(packagelist)
            i = 0
        if i > num:
            break
        if len(packagelist[0]) > 4 and packagelist[0][-4:] not in [".deb"]:
             del packagelist[0]
             continue
        i = i + 1
        package_absdir = packagedir + '/' + packagelist[0]
        if os.path.isfile(package_absdir):
            if os.system(r'sudo dpkg -i ' + package_absdir):
                print(r'dpkg -i %s failed!' % packagelist[0])
                packagelist.append(packagelist[0])
            else:
                print(r'dpkg -i  %s successeded!' % packagelist[0])
            del packagelist[0]
            print('len(packagelist): %d' % len(packagelist))

if __name__ == '__main__':
    batchinstall(os.getcwd())
