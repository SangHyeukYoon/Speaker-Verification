from librispeech import LibriSpeechDataset
from utils import preprocess_instances, BatchPreProcessor

training_set = ['train-clean-360']

train = LibriSpeechDataset(training_set, 3, pad=True)
batch_preprocessor = BatchPreProcessor('siamese', preprocess_instances(4))

([batch_x_1, batch_x_2], batch_y) = batch_preprocessor(train.build_verification_batch(12))
