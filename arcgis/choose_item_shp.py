import sys
import os
import arcpy
import pdb
from arcpy import env
import random
env.workspace = r"D:\NGCC\zheng\men"


def create_choose_dict(xy_list, threshold, corr_index=1, sor_index=0):
    xy_dict = {}
    del_num = 0
    for xy in xy_list:
        corr = xy[corr_index][sor_index]
        area = xy[2]
        if area < 0.1 or abs(corr) < abs(int(corr)) + threshold[0] or corr > abs(int(corr)) + threshold[1]:
            del_num += 1
            continue
        key = str(int(xy[corr_index][sor_index]))
        xy_dict.setdefault(key, [])
        xy_dict[key].append(xy[:2])
    print('del_num: %d' % (del_num))
    for key, value in xy_dict.items():
        xy_dict[key].sort(key=lambda item: item[corr_index][1])
    return xy_dict


def choose_num_list(dict_info, num):
    interval = 0.2
    num_list = []

    def test_choose():
        for key, value in dict_info.items():
            start = dict_info[key][0][1][1]
            for coor in dict_info[key]:
                # print(coor)
                if coor[1][1] >= start:
                    num_list.append(coor)
                    # print('append')
                    start = interval + coor[1][1]
    while abs(len(num_list) - num) > num * 0.02:
        num_list = []
        test_choose()
        print('num_list:%d,interval:%f' % (len(num_list), interval))
        interval = interval * float(len(num_list)) / float(num)*random.uniform(0.8,1.2)
    return num_list


def random_get_feature(input_shp,out_path,out_name, get_num):
    num = 0
    count = 0
    # create new shp
    # Set local variables
    out_path #= r"D:\NGCC\zheng\men"
    # out_name = "randomchoose.shp"
    geometry_type = "POLYGON"
    template = input_shp
    has_m = "DISABLED"
    has_z = "DISABLED"
    # Use Describe to get a SpatialReference object
    spatial_reference = arcpy.Describe(
        r"D:\NGCC\zheng\men\complete_within.shp").spatialReference
    # Execute CreateFeatureclass
    if os.path.exists('%s/%s' % (out_path, out_name)):
        os.remove('%s/%s' % (out_path, out_name))
    arcpy.CreateFeatureclass_management(
        out_path, out_name, geometry_type, template, has_m, has_z, spatial_reference)
    os.remove('%s/%s.xml' % (out_path, out_name))
    print('create new feature!')
    # get field name of shapefile
    fieldList = [f.name for f in arcpy.ListFields(input_shp)]
    fieldList[1] = 'SHAPE@'
    polygon_list = []
    for row in arcpy.da.SearchCursor(input_shp, ['OID@', 'SHAPE@XY', 'SHAPE@AREA']):
        num += 1
        polygon_list.append(row)
    print('All: %d' % (len(polygon_list)))
    cor_dict = create_choose_dict(
        polygon_list, (0.1, 0.95), corr_index=1, sor_index=0)
    coor_list = choose_num_list(cor_dict, get_num)
    coor_list.sort(key=lambda item: item[0])
    print('coor_list:%d' % (len(coor_list)))
    data = arcpy.da.SearchCursor(input_shp, field_names=fieldList)
    new_shp = arcpy.da.InsertCursor(
        '%s/%s' % (out_path, out_name), field_names=fieldList)
    for row in data:
        if count < len(coor_list) and row[0] == coor_list[count][0]:
            new_shp.insertRow(row)
            count += 1
    print("finished")


def select_feature_from_points(point_file, input_shp, append_shape):
    # create new shp
    # Set local variables
    # out_path = r"D:\NGCC\zheng\men"
    # out_name = "append01.shp"
    # geometry_type = "POLYGON"
    # template = input_shp
    # has_m = "DISABLED"
    # has_z = "DISABLED"
    # # Use Describe to get a SpatialReference object
    # spatial_reference = arcpy.Describe(
    #     r"D:\NGCC\zheng\men\complete_within.shp").spatialReference
    # # Execute CreateFeatureclass
    # if os.path.exists('%s/%s' % (out_path, out_name)):
    #     os.remove('%s/%s' % (out_path, out_name))
    # arcpy.CreateFeatureclass_management(
    #     out_path, out_name, geometry_type, template, has_m, has_z, spatial_reference)
    # os.remove('%s/%s.xml' % (out_path, out_name))
    # print('create new feature %s!' % (out_name))
    fieldList = [f.name for f in arcpy.ListFields(input_shp)]
    fieldList[1] = 'SHAPE@'
    data = arcpy.da.SearchCursor(input_shp, field_names=fieldList)
    new_shp = arcpy.da.InsertCursor(append_shape, field_names=fieldList)
    point_list = []
    with open(point_file) as fn:
        for point in fn.readlines():
            # info = int(point[:-1])
            info = point[:-1]
            point_list.append(info)
    print('add %d points!' % (len(point_list)))
    count = 0
    for row in data:
        if row[2] in point_list:
            new_shp.insertRow(row)
            count += 1
    print('add %d points actually!' % (count))
    print('finished!')


def main():
    print('main process')
    input_shp = r'D:\NGCC\zheng\20181129\result\Australia.shp'
    outpath = r'D:\NGCC\zheng\20181129\result'
    out_name = 'target_220.shp'
    # append_file = r'D:\NGCC\zheng\men\append01.txt'
    # append_shape = r'D:\NGCC\zheng\men\randomchoose.shp'
    random_get_feature(input_shp,outpath,out_name, 220)
    # select_feature_from_points(append_file, input_shp, append_shape)
if __name__ == '__main__':
    main()

