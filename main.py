#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
#  main.py
#
#  Copyright 2022 mc <mc@PyBuntu>
# import contextlib
import sys
import os
# import mimetypes
from PIL import Image, ImageFilter
from pyzbar import pyzbar
# from progress.bar import Bar
from progress.bar import IncrementalBar

image_ext_list = list(
    [".BMP", ".DDS", ".DIB", ".EPS", ".GIF", ".ICNS", ".ICO", ".IM", ".JPEG", ".JPG", ".MSP", ".PCX", ".PNG",
     ".PPM", ".SGI", ".SPIDER", ".TGA", ".TIFF", ".WEBP", ".XBM"])


def get_barcode(fullname):
    barcodes = list()
    if os.path.exists(fullname) and os.path.isfile(fullname):
        filename = os.path.basename(fullname)
        name, ext = os.path.splitext(filename)
        try:
            with Image.open(fullname) as im:
                barcodes = pyzbar.decode(im)
                if not (len(barcodes) > 0):
                    sharp_im = im.filter(ImageFilter.SHARPEN)
                    barcodes = pyzbar.decode(sharp_im)
        except OSError as os_error:
            print("Error open image as: {} because: {}".format(fullname, os_error))
            return barcodes
        except ValueError as val_error:
            print("Format could not be determined from the file as: {} because: {}".format(fullname, val_error))
            return barcodes
        # except:
        #     print("Another error or warning: {}".format(fullname))
        #     return barcodes
    else:
        print("No {} exist or it is no file".format(fullname))
        return barcodes

    return barcodes


def get_walks(path):
    """
    Return list of image files from dir
    :param path: maindir
    :return: list of files in maindir
    """
    filelist = []
    if os.path.exists(path) and os.path.isdir(path):
        # print("Check {} ...".format(path)
        for dirpath, dirnames, filenames in os.walk(path):
            for f in filenames:
                # image_file = os.path.join(dirpath, f)
                # check_type = mimetypes.guess_type(image_file)[0]
                # if not check_type is None:
                #   print(check_type.split("/")[0])
                #   if "image" == check_type.split("/")[0]:
                name, ext = os.path.splitext(f)
                if ext.upper() in image_ext_list:
                    filelist.append(os.path.join(dirpath, f))
    else:
        print("No {} exist or it is no dir".format(path))

    return filelist


def main(args):
    """
    Convert all found images in setted dir and it`s subdir
    :param args: set dir or empty if dir is current workdir
    :return: 0 if no errors
    """
    if len(args) < 2:
        path = os.getcwd()
    else:
        path = args[1]

    if '~' in path:
        fullpath = os.path.expanduser(path)
    else:
        fullpath = os.path.realpath(path)

    file_list = get_walks(fullpath)
    if len(file_list) > 0:
        bar = IncrementalBar('Files read', max=len(file_list))
        # bar = Bar('Countdown', max=len(file_list))
        out_list = list()
        unresolved = 0
        for image_file in file_list:
            dirname, filename = os.path.split(image_file)
            relpath = dirname.replace(fullpath, "")
            barcodes = get_barcode(image_file)
            i = 0
            if len(barcodes) > 0:
                # print("File {} barcode Found".format(image_file))
                for barcode in barcodes:
                    # print("barcode as: {}".format(barcode.data))
                    i += 1
                    out_list.append("{}\t{}\t{}\t{}\n".format(filename, relpath.replace("/\\", ""), i,
                                                              barcode.data.decode("utf-8")))
            else:
                # print("File {} barcode Not Found".format(image_file))
                out_list.append("{}\t{}\t{}\t{}\n".format(filename, relpath.replace("/\\", ""), i, ""))
                unresolved += 1
            # with open("error.log", "a") as error_file:
            # with contextlib.redirect_stderr(stderr_default):
            bar.next()
        bar.finish()
        print("There are {} unresolved files".format(unresolved))
        try:
            with open(os.path.join(fullpath, 'foundcodes.csv'), 'w', encoding='utf-8') as outfile:
                outfile.writelines(out_list)
        except OSError as os_error:
            print(
                "Error operate with file as: {} because: {}".format(os.path.join(fullpath, 'foundcodes.csv'), os_error))
    else:
        print("No files found for path as: {}".format(fullpath))

    return 0


if __name__ == '__main__':
    # stderr_default = sys.stderr
    sys.exit(main(sys.argv))
