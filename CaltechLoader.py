from paz.abstract import Loader


class DictLoader(Loader):
    """
    Loader initialized from a list of dictionaries.
    """

    def __init__(self, img_list, path=None, split="test", class_names=('person', 'people'), name="CALTECH"):
        """
        Initializes Loader from list.
        :param img_list: List of images/boxes in the format [{img_path,bbox_vectors},...]
        :param name: Name of the dataset.
        """
        super().__init__(path, split, class_names, name)
        self.img_dict = self.list_to_dict(img_list)

    def list_to_dict(self, img_list):
        img_dict = {}
        for d in img_list:
            img_dict[d["image"]] = d["boxes"]  # convert format
        return img_dict

    def load_data(self):
        """
        Essentially a getter that returns the stored image dictionary.
        :return: image dictionary
        """
        return self.img_dict
