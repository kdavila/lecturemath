
import xml.etree.ElementTree as ET

class UniqueWordGroup:
    def __init__(self, start_word, start_frame):
        # ending frame can be computed by adding length of word_refs to start minus 1
        # as all Words must appear in contiguous frames.
        self.words_refs = [start_word]
        self.start_frame = start_frame

    def lastFrame(self):
        return self.start_frame + len(self.words_refs) - 1

    def n_frames(self):
        return len(self.words_refs)

    def strID(self):
        return str(self.start_frame) + "-" + UniqueWordGroup.wordID(self.words_refs[0])

    def __eq__(self, other):
        if not isinstance(other, UniqueWordGroup):
            return False
        else:
            return self.words_refs == other.words_refs

    @staticmethod
    def wordID(word):
        return "-".join([str(dim) for dim in word])

    @staticmethod
    def GroupsFromXML(all_kf_words, xml_filename, namespace=''):
        # returns the groups and the inverted index ...
        # also, input set of key-frames words for consistency validation!

        # Initially, build inverted indexes for groups and key-frames Words
        word_group = []
        word_index = []
        for kf_words in all_kf_words:
            group_dict = {}
            index_dict = {}
            for word in kf_words.get_words():
                word_id = UniqueWordGroup.wordID(word)
                group_dict[word_id] = None
                index_dict[word_id] = word

            word_group.append(group_dict)
            word_index.append(index_dict)

        ids_added = [[] for kf_words in all_kf_words]
        ids_removed = [[] for kf_words in all_kf_words]
        ids_file = [{} for kf_words in all_kf_words]

        # load file!
        tree = ET.parse(xml_filename)
        root = tree.getroot()
        video_words_root = root.find(namespace + 'VideoWords')
        kf_words_xml_roots = video_words_root.findall(namespace + 'KeyFrameWords')

        # for key-frame words element
        for kf_idx, xml_kf_words in enumerate(kf_words_xml_roots):
            all_words_root = xml_kf_words.find(namespace + 'Words')
            word_roots = all_words_root.findall(namespace + 'Word')

            # read words recorded in file ...
            for xml_word in word_roots:
                word_id = xml_word.text.strip()
                ids_file[kf_idx][word_id] = True

                # mark those which are not currently available in the provided segmentation
                # (element in XML file, not in Word segmentation)
                if not word_id in word_index[kf_idx]:
                    print("Key-frame # " + str(all_kf_words[kf_idx].kf_annotation.idx) + ", missing Word {" + word_id + "}")
                    ids_removed[kf_idx].append(word_id)

            # now, check for new words ...
            # (element not int XML file, from Word Segmentation)
            for kf_word_id in word_index[kf_idx]:
                if not kf_word_id in ids_file[kf_idx]:
                    print("Key-frame # " + str(all_kf_words[kf_idx].kf_annotation.idx) + ", Added Word {" + kf_word_id + "}")
                    ids_added[kf_idx].append(kf_word_id)

        print("Total Missing: " + str(sum([len(words_missing) for words_missing in ids_removed])))
        print("Total Added: " + str(sum([len(words_added) for words_added in ids_added])))

        unique_groups = []
        groups_root = root.find(namespace + 'WordGroups')
        groups_xml_roots = groups_root.findall(namespace + 'WordGroup')
        for group_xml in groups_xml_roots:
            group_start = int(group_xml.find(namespace + "Start").text.strip())

            group_words_root = group_xml.find(namespace + "Words")
            group_word_xml_roots = group_words_root.findall(namespace + "Word")

            valid_group_ids = []

            for kf_offset, group_word_xml in enumerate(group_word_xml_roots):
                word_id = group_word_xml.text.strip()

                if word_id in word_group[group_start + kf_offset]:
                    valid_group_ids.append(word_id)
                else:
                    # mismatch found, stop loading group
                    break

            if len(valid_group_ids) > 0:
                # create group and link with valid members ...
                first_id = valid_group_ids[0]
                new_group = UniqueWordGroup(word_index[group_start][first_id], group_start)
                # first member
                word_group[group_start][first_id] = new_group

                # The rest of the members
                for kf_offset in range(1, len(valid_group_ids)):
                    # add to the group
                    new_group.words_refs.append(word_index[group_start + kf_offset][valid_group_ids[kf_offset]])
                    # link to the group
                    word_group[group_start + kf_offset][valid_group_ids[kf_offset]] = new_group

                # add to the general set
                unique_groups.append(new_group)

        # find Words without groups

        for kf_idx in range(len(all_kf_words)):
            for word_id in word_group[kf_idx]:
                if word_group[kf_idx][word_id] is None:
                    print("Will create group for new Word {" + word_id + "} on Keyframe # " + str(all_kf_words[kf_idx].kf_annotation.idx))

                    # creating ...
                    new_group = UniqueWordGroup(word_index[kf_idx][word_id], kf_idx)
                    # add link ...
                    word_group[kf_idx][word_id] = new_group
                    # add group
                    unique_groups.append(new_group)

        print("Loaded: " + str(len(unique_groups)) + " Word groups (Unique Words)")

        return word_group, unique_groups

    @staticmethod
    def GenerateGroupsXML(video_kf_words, groups):
        xml_str = "<UniqueWords>\n"

        # first add the complete set of ccs per key-frames, using ID
        xml_str += "  <VideoWords>\n"
        for kf_words in video_kf_words:
            kf_xml = "    <KeyFrameWords>\n"
            kf_xml += "      <Words>\n"
            for word in kf_words.get_words():
                kf_xml += "         <Word>" + UniqueWordGroup.wordID(word) + "</Word>\n"
            kf_xml += "      </Words>\n"
            kf_xml += "    </KeyFrameWords>\n"
            xml_str += kf_xml
        xml_str += "  </VideoWords>\n"

        # Then, add the group information
        xml_str += "  <WordGroups>\n"
        for group in groups:
            xml_str += "    <WordGroup>\n"
            xml_str += "        <Start>" + str(group.start_frame) + "</Start>\n"
            xml_str += "        <End>" + str(group.start_frame + len(group.words_refs) - 1) + "</End>\n"
            xml_str += "        <Words>\n"
            for word in group.words_refs:
                xml_str += "          <Word>" + UniqueWordGroup.wordID(word) + "</Word>\n"
            xml_str += "        </Words>\n"
            xml_str += "    </WordGroup>\n"
        xml_str += "  </WordGroups>\n"

        xml_str += "</UniqueWords>\n"

        return xml_str

    @staticmethod
    def Copy(original):
        copied = UniqueWordGroup(None, original.start_frame)
        copied.words_refs = list(original.words_refs)

        return copied

    @staticmethod
    def Split(original, split_frame):
        offset_split = split_frame - original.start_frame

        if offset_split <= 0 or offset_split >= len(original.words_refs):
            # nothing to split ...
            return None
        else:
            # create new Unique Word group ...
            new_group = UniqueWordGroup(None, split_frame)
            new_group.words_refs = list(original.words_refs[offset_split:])

            # limit the original list
            original.words_refs = original.words_refs[:offset_split]

            return new_group

