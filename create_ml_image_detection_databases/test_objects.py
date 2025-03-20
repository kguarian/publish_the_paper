from json_objects import AreaSelection, Annotations, TrainingSample
import logging
import json

logging.basicConfig(level=logging.DEBUG)
mylogger = logging.getLogger()


class TestObjects():
    def test_areaSelection(self):
        area_sel = AreaSelection(x=0, y=1, width=2, height=3)
        filename = 'test_area_selectoin.json'
        with open(filename, 'w') as file:
            file.write(area_sel.toJSON())
            file.close()
        assert True

    def test_annotation(self):
        area_sel = AreaSelection(x=0, y=1, width=2, height=3)
        data_annot = Annotations(label='test', coordinates=area_sel)
        filename = 'test_annotation.json'
        with open(filename, 'w') as file:
            file.write(data_annot.toJSON())
            file.close()
        assert True

    def test_training_sample(self):
        area_sel = AreaSelection(x=0, y=1, width=2, height=3)
        data_annot = Annotations(label='test', coordinates=area_sel)
        training_sample = TrainingSample(
            filename='test.png', annotation=data_annot)
        filename = 'test_training_sample.json'
        with open(filename, 'w') as file:
            file.write(training_sample.toJSON())
            file.close()
        assert True

    def test_list_training_samples(self):
        X = [0, 1, 2, 3]
        Y = [4, 5, 6, 7]
        width = [8, 9, 10, 11]
        height = [12, 13, 14, 15]
        labels = ['sig_a', 'sig_b', 'sig_c', 'sig_d']
        training_samples = []
        for i in range(4):
            area_sel = AreaSelection(
                x=X[i], y=Y[i], width=width[i], height=height[i])
            data_annot = Annotations(label=labels[i], coordinates=area_sel)
            training_sample = TrainingSample(
                filename='test_'+str(i)+'.png', annotation=data_annot)
            training_samples.append(training_sample)
        filename = 'test_list_training_samples.json'
        with open(filename, 'w') as file:
            out = json.dumps(
                training_samples,
                default=lambda o: o.__dict__,
                sort_keys=True,
                indent=4
            )
            file.write(out)
