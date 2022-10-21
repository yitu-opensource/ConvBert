# coding=utf-8

"""Preprocessess the Open WebText corpus for pre-training."""

import argparse
import multiprocessing
import os
import random
import tarfile
import time
import tensorflow.compat.v1 as tf

import build_pretraining_dataset
from util import utils


def write_examples(job_id, args):
  """A single process creating and writing out pre-processed examples."""
  job_tmp_dir = os.path.join(args.data_dir, "tmp", "job_" + str(job_id))
  owt_dir = os.path.join(args.data_dir, "openwebtext")

  def log(*args):
    msg = " ".join(map(str, args))
    print("Job {}:".format(job_id), msg)

  log("Creating example writer")
  example_writer = build_pretraining_dataset.ExampleWriter(
      job_id=job_id,
      vocab_file=os.path.join(args.data_dir, "vocab.txt"),
      output_dir=os.path.join(args.data_dir, "pretrain_tfrecords"),
      max_seq_length=args.max_seq_length,
      num_jobs=args.num_processes,
      blanks_separate_docs=False,
      strip_accents=args.strip_accents,
  )
  log("Writing tf examples")
  fnames = sorted(tf.io.gfile.listdir(owt_dir))
  fnames = [f for (i, f) in enumerate(fnames)
            if i % args.num_processes == job_id]
  random.shuffle(fnames)
  start_time = time.time()
  for file_no, fname in enumerate(fnames):
    if file_no > 0 and file_no % 10 == 0:
      elapsed = time.time() - start_time
      log("processed {:}/{:} files ({:.1f}%), ELAPSED: {:}s, ETA: {:}s, "
          "{:} examples written".format(
              file_no, len(fnames), 100.0 * file_no / len(fnames), int(elapsed),
              int((len(fnames) - file_no) / (file_no / elapsed)),
              example_writer.n_written))
    utils.rmkdir(job_tmp_dir)
    with tarfile.open(os.path.join(owt_dir, fname)) as f:
      def is_within_directory(directory, target):
          
          abs_directory = os.path.abspath(directory)
          abs_target = os.path.abspath(target)
      
          prefix = os.path.commonprefix([abs_directory, abs_target])
          
          return prefix == abs_directory
      
      def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
      
          for member in tar.getmembers():
              member_path = os.path.join(path, member.name)
              if not is_within_directory(path, member_path):
                  raise Exception("Attempted Path Traversal in Tar File")
      
          tar.extractall(path, members, numeric_owner=numeric_owner) 
          
      
      safe_extract(f, job_tmp_dir)
    extracted_files = tf.io.gfile.listdir(job_tmp_dir)
    random.shuffle(extracted_files)
    for txt_fname in extracted_files:
      example_writer.write_examples(os.path.join(job_tmp_dir, txt_fname))
  example_writer.finish()
  log("Done!")


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--data-dir", required=True,
                      help="Location of data (vocab file, corpus, etc).")
  parser.add_argument("--max-seq-length", default=128, type=int,
                      help="Number of tokens per example.")
  parser.add_argument("--num-processes", default=1, type=int,
                      help="Parallelize across multiple processes.")

  # toggle strip-accents and set default to True which is the default behavior
  parser.add_argument("--do-strip-accents", dest='strip_accents',
                      action='store_true', help="Strip accents (default).")
  parser.add_argument("--no-strip-accents", dest='strip_accents',
                      action='store_false', help="Don't strip accents.")
  parser.set_defaults(strip_accents=True)

  args = parser.parse_args()

  utils.rmkdir(os.path.join(args.data_dir, "pretrain_tfrecords"))
  if args.num_processes == 1:
    write_examples(0, args)
  else:
    jobs = []
    for i in range(args.num_processes):
      job = multiprocessing.Process(target=write_examples, args=(i, args))
      jobs.append(job)
      job.start()
    for job in jobs:
      job.join()


if __name__ == "__main__":
  main()
