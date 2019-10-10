import argparse
import functools
import glob
import os
import string
import subprocess
import time
from collections import Counter
from random import randint

import yaml
from create_csv import create_csv

file_counter = Counter()


def retry(num_times=3, sleep=60, exception_type=Exception, log=None):
    def decorator_retry(func):
        @functools.wraps(func)
        def wrapper_retry(*args, **kwargs):
            for count in range(num_times):
                try:
                    return func(*args, **kwargs)
                except exception_type as e:
                    if log:
                        log("retry")
                        log(e)
                    if count >= num_times - 1:
                        if log:
                            log("throwing")
                        raise e
                    time.sleep(sleep)

        return wrapper_retry

    return decorator_retry


@retry(
    num_times=10, sleep=randint(600, 3600), exception_type=subprocess.CalledProcessError, log=print
)
def download_video(source, output_path):
    # "--retries 10",
    # "--ignore-errors",
    # f"--max-downloads {max_downloads}",
    command = " ".join(
        [
            "youtube-dl",
            "--force-ipv4",
            "--download-archive downloaded.txt",
            "--no-post-overwrites",
            "--continue",
            "--no-overwrites",
            "--sleep-interval 20",
            "--max-sleep-interval 60",
            "--extract-audio",
            "--audio-format wav",
            f"{source}",
            f'-o "{output_path}/%(title)s.%(ext)s"',
        ]
    )

    subprocess.check_output(command, shell=True)


def get_video_list(source: str) -> list:
    video_list = subprocess.check_output(
        " ".join(
            [
                "youtube-dl",
                "--dump-json",
                "--flat-playlist",
                f"{source}",
                "| jq -r '.id'",
                "| sed 's_^_https://youtu.be/_'",
            ]
        ),
        shell=True,
    )

    return video_list.decode("utf-8").strip().split("\n")


def read_yaml(file_name):
    with open(file_name, "r") as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def clean_filename(filename):
    valid_chars = "-_%s%s" % (string.ascii_letters, string.digits)
    new_name = "".join(c for c in filename if c in valid_chars)
    new_name = new_name.replace(" ", "_")
    return new_name


def download(language, source, source_name, source_type):

    output_path_raw = os.path.join(args.output_path, "raw", language, source_name)

    if source_type == "playlist":
        playlist_archive = os.path.join(output_path_raw, "archive.txt")

        print("Downloading {0} {1} to {2}".format(source_type, source_name, output_path_raw))
        command = """youtube-dl -i --download-archive {} --max-filesize 50m --no-post-overwrites --max-downloads {} --sleep-interval 5 --max-sleep-interval 60 --extract-audio --audio-format wav {} -o "{}/%(title)s.%(ext)s" """.format(
            playlist_archive, args.max_downloads, source, output_path_raw
        )
        subprocess.call(command, shell=True)
    else:
        os.makedirs(output_path_raw, exist_ok=True)
        already_down = len(os.listdir(output_path_raw))

        if os.path.exists(output_path_raw) and already_down >= int(args.max_downloads):
            print(f"skipping {output_path_raw} because the target folder already exists")
        else:
            print("Downloading {0} {1} to {2}".format(source_type, source_name, output_path_raw))

            video_list = get_video_list(source)

            for video in video_list[already_down : int(args.max_downloads)]:
                download_video(video, output_path_raw)

    # Use ffmpeg to convert and split WAV files into 10 second parts
    output_path_segmented = os.path.join(args.output_path, "segmented", language, source_name)
    segmented_files = glob.glob(os.path.join(output_path_segmented, "*.wav"))

    if source_type == "playlist" or not os.path.exists(output_path_segmented):
        os.makedirs(output_path_segmented, exist_ok=True)

        files = glob.glob(os.path.join(output_path_raw, "*.wav"))

        for f in files:

            cleaned_filename = clean_filename(os.path.basename(f))
            cleaned_filename = cleaned_filename[:-4]

            if source_type == "playlist":
                waves = [f for f in segmented_files if cleaned_filename in f]
                if len(waves) > 0:
                    continue

            output_filename = os.path.join(output_path_segmented, cleaned_filename + "_%03d.wav")

            command = [
                "ffmpeg",
                "-y",
                "-i",
                f,
                "-map",
                "0",
                "-ac",
                "1",
                "-ar",
                "16000",
                "-f",
                "segment",
                "-segment_time",
                "10",
                output_filename,
            ]
            subprocess.call(command)

    file_counter[language] += len(glob.glob(os.path.join(output_path_segmented, "*.wav")))


def download_user(language, user):
    user_selector = "ytuser:%s" % user
    download(language, user_selector, user, "user")


def download_playlist(language, playlist_name, playlist_id):
    download(language, playlist_id, playlist_name, "playlist")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", dest="output_path", default=os.getcwd(), required=True)
    parser.add_argument("--downloads", dest="max_downloads", default=1200)
    args = parser.parse_args()

    sources = read_yaml("sources.yml")
    for language, categories in sources.items():
        for user in categories["users"]:
            if user is None:
                continue

            download_user(language, user)

        for category in categories["playlists"]:
            if category is None:
                continue

            playlist_name = category
            playlist_id = category
            download_playlist(language, playlist_name, playlist_id)

    create_csv(os.path.join(args.output_path, "segmented"))

    print(file_counter)
