#!/usr/bin/env python3

import os
import tqdm
import subprocess
import argparse


def convert_pdf_to_txt(pdf_path: str, txt_path: str, overwrite=False):
    """
    Convert a pdf file to a txt file.
    """
    if not os.path.exists(txt_path) or overwrite:
        subprocess.call(['pdftotext', pdf_path, txt_path])


def main():
    """
    Main function.
    """
    parser = argparse.ArgumentParser(
        description='Convert PDF to TXT.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--pdf_dir',
        help='Path to the PDF file directory.',
        type=str,
        default='./pdf')
    parser.add_argument(
        '--txt_dir',
        help='Path to the TXT file directory.',
        type=str,
        default='./txt')
    parser.add_argument(
        '--overwrite',
        help='Overwrite existing TXT files.',
        action='store_true',
        default=False)

    args = parser.parse_args()
    parser.parse_args(namespace=args)
    print(args)

    pdf_dir = args.pdf_dir
    txt_dir = args.txt_dir
    overwrite = args.overwrite
    # Create the txt directory if it does not exist
    if not os.path.exists(txt_dir):
        os.mkdir(txt_dir)
    # Loop through the pdfs
    q = tqdm.tqdm(os.listdir(pdf_dir))
    for pdf_file in q:
        # Get the path to the pdf
        pdf_path = os.path.join(pdf_dir, pdf_file)
        # Get the path to the txt
        txt_path = os.path.join(txt_dir, pdf_file.replace('.pdf', '.txt'))
        # Convert the pdf to txt
        convert_pdf_to_txt(pdf_path, txt_path, overwrite)
        # Update the progress bar
        q.set_description(f'Processing {pdf_file}')


if __name__ == '__main__':
    main()
