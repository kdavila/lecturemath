
import xml.etree.ElementTree as ET
from .lecture_info import LectureInfo
from .indexing_info import IndexingInfo

class MetaDataDB:
    Namespace = ''

    def __init__(self, name):
        self.name = name

        self.output_temporal = ""
        self.output_preprocessed = ""
        self.output_indices = ""
        self.output_images = ""
        self.output_videos = ""
        self.output_annotations = ""
        self.output_summaries = ""
        self.output_search_results = ""

        self.lectures = []
        self.datasets = {}

        self.indexing = None

    @staticmethod
    def get_text_or_default(xml_node, tag_name, default):
        sub_element = xml_node.find(MetaDataDB.Namespace + tag_name)
        if sub_element is not None:
            return sub_element.text
        else:
            return default

    @staticmethod
    def from_XML_node(root):
        # read the most general metadata
        data = root.find(MetaDataDB.Namespace + 'DataBase')
        name = data.find(MetaDataDB.Namespace + 'Name').text

        outputs = data.find(MetaDataDB.Namespace + 'OutputPaths')
        output_temporal = MetaDataDB.get_text_or_default(outputs, 'Temporal', None)
        output_preprocessed = MetaDataDB.get_text_or_default(outputs, 'Preprocessed', None)
        output_indices = MetaDataDB.get_text_or_default(outputs, 'Indices', None)
        output_images = MetaDataDB.get_text_or_default(outputs, 'Images', None)
        output_videos = MetaDataDB.get_text_or_default(outputs, 'Videos', None)
        output_annotations = MetaDataDB.get_text_or_default(outputs, 'Annotations', None)
        output_summaries = MetaDataDB.get_text_or_default(outputs, 'Summaries', None)
        output_search_results = MetaDataDB.get_text_or_default(outputs, 'SearchResults', None)

        # create DB object
        db = MetaDataDB(name)
        db.output_temporal = output_temporal
        db.output_preprocessed = output_preprocessed
        db.output_indices = output_indices
        db.output_images = output_images
        db.output_videos = output_videos
        db.output_annotations = output_annotations
        db.output_summaries = output_summaries
        db.output_search_results = output_search_results

        # now, read every lecture info in Database
        lectures = data.find(MetaDataDB.Namespace + "Lectures")
        for lecture_node in lectures.findall(MetaDataDB.Namespace + "Lecture"):
            # ... read XML
            lecture = LectureInfo.from_XML_node(lecture_node)
            # .... add
            db.lectures.append(lecture)

        # check if datasets are defined ...
        datasets = data.find(MetaDataDB.Namespace + 'Datasets')

        for node in datasets:
            dataset_name = node.tag.lower()

            ds_lectures = []
            lecture_titles_xml = node.findall(MetaDataDB.Namespace + 'LectureTitle')
            for xml_title in lecture_titles_xml:
                lecture = db.get_lecture(xml_title.text)
                ds_lectures.append(lecture)

            db.datasets[dataset_name] = ds_lectures

        # check for indexing information ...
        indexing_root = data.find(MetaDataDB.Namespace + 'LectureIndexing')
        if indexing_root:
            db.indexing = IndexingInfo.from_XML_node(indexing_root)

        return db

    def get_lecture(self, title):
        title = title.lower()
        current_lecture = None
        for lecture in self.lectures:
            if lecture.title.lower() == title:
                current_lecture = lecture
                break

        return current_lecture

    def get_dataset(self, name):
        key = name.lower()
        if key in self.datasets:
            return self.datasets[key]
        else:
            return None

    def get_lectures(self, title_filter, all_on_empty=False):
        if title_filter is None:
            return self.lectures
        else:
            candidates = [lect for lect in self.lectures if lect.title[:len(title_filter)].lower() == title_filter.lower()]
            if all_on_empty and len(candidates) == 0:
                return self.lectures
            else:
                return candidates

    @staticmethod
    def from_file(filename):
        tree = ET.parse(filename)
        root = tree.getroot()

        return MetaDataDB.from_XML_node(root)

    @staticmethod
    def load_database_lecture(database_filename, lecture_name):
        try:
            database = MetaDataDB.from_file(database_filename)
        except:
            print("Invalid database file")
            return None, None

        current_lecture = database.get_lecture(lecture_name)

        if current_lecture is None:
            print("Lecture not found in database")
            print("Available lectures:")

            candidate_lectures = database.get_lectures(lecture_name, True)

            tempo_str = ""
            for idx, lecture in enumerate(candidate_lectures):
                tempo_str += lecture.title + ("\t" if (idx + 1) % 4 > 0 else "\n")
            print(tempo_str)

            return None, None

        return database, current_lecture