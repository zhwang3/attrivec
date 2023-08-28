import pandas as pd
import numpy as np
import pickle
import argparse


class AttributeOperator:
    def __init__(self, dictlist):
        self.dictlist = dictlist
        self.keys = []
        self.string_attributes = {}

    def build_dict_keys(self, dictitem):
        keys = []
        for i in dictitem.items():
            keys.append(i[0])
            if type(i[1]) is str:
                if type(eval(i[1])) is dict:
                    keys.extend(self.build_dict_keys(eval(i[1])))
                    keys.remove(i[0])
        return list(set(keys))

    def get_matrix(self):
        dim = len(self.keys)
        size = len(self.dictlist)
        self.attribute_matrix = np.zeros((size, dim))

    def retrieve_keys(self):
        keys = []
        for i in self.dictlist:
            if i is not None:
                keys.extend(self.build_dict_keys(i))
        self.keys = list(set(keys))

    def numerized_attribute(self, attribute_value, attribute):
        if type(attribute_value) is bool:
            if attribute_value:
                return 1
            else:
                return 0
        elif type(attribute_value) is int or type(attribute_value) is float:
            return attribute_value
        elif type(attribute_value) is str:
            if attribute in self.string_attributes.keys():
                if attribute_value not in self.string_attributes[attribute].keys():
                    self.string_attributes[attribute][attribute_value] = max(
                        self.string_attributes[attribute].values()) + 1
            else:
                self.string_attributes.update({attribute:{attribute_value:1}})
            return self.string_attributes[attribute][attribute_value]

    def set_attribute_vector(self, dictitem, mat_row):
        for i in dictitem.items():
            if type(i[1]) is str:
                if type(eval(i[1])) is dict:
                    self.set_attribute_vector(eval(i[1]),mat_row)
                else:
                    att_value = self.numerized_attribute(eval(i[1]), i[0])
                    self.attribute_matrix[mat_row, self.keys.index(i[0])] = att_value
            else:
                att_value = self.numerized_attribute(i[1], i[0])
                self.attribute_matrix[mat_row, self.keys.index(i[0])] = att_value

    def build_attribute_matrix(self):
        mat_row = 0
        for i in self.dictlist:
            if i is not None:
                self.set_attribute_vector(i, mat_row)
            else:
                self.attribute_matrix[mat_row, :] = np.zeros([1, len(self.keys)])
            mat_row += 1

if __name__ == '__main__':
    parser=argparse.ArgumentParser(prog='Yelp Dataset Business Attributes Vectorize Tool',
                            description='Yelp stores business attributes as dict in their open dataset. However, we need attributes in matrix format in some scenarios such as deep learning. This tool can help to achieve this.',
                            )
    parser.add_argument('datafile', default='dataset/yelp_academic_dataset_business.json')
    parser.add_argument('output_location', default='.')

    args=parser.parse_args()

    business_raw = pd.read_json(args.datafile, lines=True, chunksize=1000)
    business_raw = pd.concat(business_raw)


    ap = AttributeOperator(business_raw['attributes'])
    ap.retrieve_keys()
    ap.get_matrix()
    ap.build_attribute_matrix()

    with open(args.output_location+'/attributes_nparray.pickle','wb') as f:
        pickle.dump(ap.attribute_matrix,f)

    with open(args.output_location+'/string_attributes_description_dict.pickle','wb') as f:
        pickle.dump(ap.string_attributes,f)
