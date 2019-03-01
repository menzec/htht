This script get the bndbox area and length-width ratio and output the result as a text file.
The result file has four columes respectively:
length,width,,area,length-width ratio

Usage:
    -x or --xmldir:the folder of xml files,
    -o or --outfile:the result file

Attention:
    If the width or length of budbox is zero,this budbox will be skiped.

此脚本文件将xml文件中的目标框的长、宽、面积、长宽比输出到一个文本文件中。
结果文件有四列，分别为：
长、宽、面积、长宽比(保留四位小数)

参数意义：
    -x --xmldir :存储xml文件的文件夹，会读取文件夹中的所有xml文件
    -o --outfile :结果文件名
