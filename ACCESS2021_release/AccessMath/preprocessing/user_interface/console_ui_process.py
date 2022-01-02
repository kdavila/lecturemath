
import os

from AM_CommonTools.configuration.configuration import Configuration
from AM_CommonTools.util.time_helper import TimeHelper
from AccessMath.data.lecture_info import LectureInfo
from AccessMath.data.meta_data_DB import MetaDataDB
from AccessMath.preprocessing.video_processor.video_processor import VideoProcessor
from AccessMath.preprocessing.video_processor.image_list_processor import ImageListProcessor
from AccessMath.util.misc_helper import MiscHelper


class ConsoleUIProcess:
    def __init__(self, database_file, optional_params, input_temp_prefix, output_temp_prefix):
        self.database_file = database_file
        self.raw_params = optional_params
        self.input_temp_prefix = input_temp_prefix
        self.output_temp_prefix = output_temp_prefix

        self.database = None
        self.params = None
        self.current_lecture = None

        self.temp_dir = None
        # self.out_dir = None # Deprecated!
        self.img_dir = None

        self.debug_max_time = 0

        self.configuration = None

    def initialize(self):
        # load database info
        try:
            self.database = MetaDataDB.from_file(self.database_file)
        except Exception as e:
            print("Invalid database file")
            print(e)
            return False

        self.params = MiscHelper.optional_parameters(self.raw_params, 0)

        # process the specified dataset(s)
        if "d" in self.params:
            if not isinstance(self.params["d"], list):
                self.params["d"] = [self.params["d"]]

            valid_datasets = []
            for name in self.params["d"]:
                dataset = self.database.get_dataset(name)

                if dataset is None:
                    print("Invalid Dataset name <" + name + ">")
                    return False
                else:
                    valid_datasets.append(dataset)

            self.params["d"] = valid_datasets

        # process only the specified lectures
        if "l" in self.params:
            if not isinstance(self.params["l"], list):
                self.params["l"] = [self.params["l"]]

            self.params["l"] = [name.lower() for name in self.params["l"]]

        # override the input prefix
        if "i" in self.params:
            self.input_temp_prefix = self.params["i"]

        # override the output prefix
        if "o" in self.params:
            self.input_temp_prefix = self.params["o"]

        if self.configuration is None:
            out_prefix = ""
        else:
            out_prefix = self.configuration.get("OUTPUT_PATH") + "/"

        self.temp_dir = out_prefix + self.database.output_temporal
        self.img_dir = out_prefix + self.database.output_images

        # success loading database file ..
        return True

    def get_lecture_params(self, lecture):
        assert isinstance(lecture, LectureInfo)

        out_file = str(lecture.id) + ".dat"

        # main video file names
        m_videos = [self.configuration.get_str("VIDEO_FILES_PATH") + "/" + video["path"] for video in lecture.main_videos]

        # check if skip lecture
        if "l" in self.params or "d" in self.params:
            # by default, skip if not in the list of specified lectures/datasets
            skip = True

            # check if in the list of lectures to include
            if "l" in self.params and lecture.title.lower() in self.params["l"]:
                # specified by name
                skip = False

            if "d" in self.params:
                for dataset in self.params["d"]:
                    if lecture in dataset:
                        # found in one of the specified datasets ..
                        skip = False
                        break
        else:
            # by default, process all lectures
            skip = False

        if skip:
            print("Skipping  <" + lecture.title + ">")
        else:
            print("Processing: <" + lecture.title + ">")

        return m_videos, out_file, skip

    def start_video_processing(self, frames_per_second, get_worker_function, get_results_function, frames_limit=0,
                               verbose=False,force_no_seek=False):
        for lecture in self.database.lectures:
            self.current_lecture = lecture
            m_videos, out_file, skip = self.get_lecture_params(lecture)

            if skip:
                continue

            # create a worker ...
            worker = get_worker_function(self)

            # execute the actual process ....
            processor = VideoProcessor(m_videos, frames_per_second)
            if "forced_width" in lecture.parameters:
                processor.force_resolution(lecture.parameters["forced_width"], lecture.parameters["forced_height"])
            processor.doProcessing(worker, frames_limit, verbose,force_no_seek) # 0

            # save results
            if self.output_temp_prefix is not None:
                results = get_results_function(worker)

                os.makedirs(self.temp_dir, exist_ok=True)
                if not isinstance(self.output_temp_prefix, list):
                    MiscHelper.dump_save(results, self.temp_dir + '/' + self.output_temp_prefix + out_file)
                else:
                    for out_idx, temp_prefix in enumerate(self.output_temp_prefix):
                        MiscHelper.dump_save(results[out_idx], self.temp_dir + '/' + temp_prefix + out_file)

    def start_input_processing(self, process_function):
        for lecture in self.database.lectures:
            self.current_lecture = lecture
            m_videos, lecture_file, skip = self.get_lecture_params(lecture)

            if skip:
                continue

            # read temporal file
            if self.input_temp_prefix is None:
                # null-input process (convenient way to process lectures)
                input_data = None
            else:
                if not isinstance(self.input_temp_prefix, list):
                    input_data = MiscHelper.dump_load(self.temp_dir + '/' + self.input_temp_prefix + lecture_file)
                else:
                    input_data = []
                    for temp_prefix in self.input_temp_prefix:
                        input_data.append(MiscHelper.dump_load(self.temp_dir + '/' + temp_prefix + lecture_file))


            # execute the actual process ....
            timer = TimeHelper()
            timer.startTimer()
            results = process_function(self, input_data)
            timer.endTimer()

            print("Process Finished in: " + timer.totalElapsedStamp())

            # save results
            if self.output_temp_prefix is not None:
                os.makedirs(self.temp_dir, exist_ok=True)
                if not isinstance(self.output_temp_prefix, list):
                    MiscHelper.dump_save(results, self.temp_dir + '/' + self.output_temp_prefix + lecture_file)
                else:
                    for out_idx, temp_prefix in enumerate(self.output_temp_prefix):
                        MiscHelper.dump_save(results[out_idx], self.temp_dir + '/' + temp_prefix + lecture_file)

    def start_image_list_preprocessing(self, get_worker_function, get_results_function, img_extension='.png',
                                       frames_limit=0, verbose=False):

        src_dir = self.configuration.get_str("OUTPUT_FRAME_EXPORT")

        for lecture in self.database.lectures:
            self.current_lecture = lecture
            _, out_file, skip = self.get_lecture_params(lecture)

            if skip:
                continue

            # create a worker ...
            worker = get_worker_function(self)

            # execute the actual process ....
            processor = ImageListProcessor('{}/{}'.format(src_dir, self.current_lecture.title),
                                           img_extension=img_extension)
            if verbose:
                print('Opening exported image folder {}{}'.format(src_dir, self.current_lecture.title))
            if "forced_width" in lecture.parameters:
                processor.force_resolution(lecture.parameters["forced_width"], lecture.parameters["forced_height"])
            processor.doProcessing(worker, frames_limit, verbose)  # 0

            # save results
            if self.output_temp_prefix is not None:
                results = get_results_function(worker)
                os.makedirs(self.temp_dir, exist_ok=True)

                if not isinstance(self.output_temp_prefix, list):
                    MiscHelper.dump_save(results, self.temp_dir + '/' + self.output_temp_prefix + out_file)
                else:
                    for out_idx, temp_prefix in enumerate(self.output_temp_prefix):
                        MiscHelper.dump_save(results[out_idx], self.temp_dir + '/' + temp_prefix + out_file)


    @staticmethod
    def usage_check(argvs):
        #usage check
        if len(argvs) < 2:
            print("Usage: python " + argvs[0] + " database [options]")
            print("Where")
            print("\tdatabase\t= Database metadata file")
            print("")
            print("Options")
            print("\t-l [lecture]\t: Process only the specified lecture(s)")
            print("\t-d [dataset_name(s)]\t: Process only the specified dataset(s)")
            return False
        else:
            return True

    @staticmethod
    def usage_with_config_check(argvs):
        # usage check
        if len(argvs) < 2:
            print("Usage: python " + argvs[0] + " config [options]")
            print("Where")
            print("\tconfig\t= AccessMath Configuration File")
            print("")
            print("Options")
            print("\t-l [lecture]\t: Process only the specified lecture(s)")
            print("\t-d [dataset_name(s)]\t: Process only the specified dataset(s)")
            return False
        else:
            return True

    @staticmethod
    def FromConfigPath(config_filename, optional_params, input_params, output_params):
        configuration = Configuration.from_file(config_filename)

        database_file = configuration.get("VIDEO_DATABASE_PATH")

        # check input/outputs (None, single value, list)
        if input_params is None:
            # zero inputs 
            input_prefixes = None
        elif isinstance(input_params, list):
            # multiple inputs
            input_prefixes = [configuration.get(param_name) for param_name in input_params]
        else:
            # assume one input
            input_prefixes = configuration.get(input_params)

        if output_params is None:
            # zero outputs
            output_prefixes = None
        elif isinstance(output_params, list):
            # multiple outputs
            output_prefixes = [configuration.get(param_name) for param_name in output_params]
        else:
            # assume one output
            output_prefixes = configuration.get(output_params)

        # check for empty process parameter list....
        if len(optional_params) == 0:
            # check for default parameters in the configuration
            if configuration.contains("DEFAULT_CONSOLE_UI_PROCESS_PARAMS"):
                optional_params = configuration.get("DEFAULT_CONSOLE_UI_PROCESS_PARAMS")

        console_ui_process = ConsoleUIProcess(database_file, optional_params, input_prefixes, output_prefixes)
        console_ui_process.configuration = configuration

        return console_ui_process
