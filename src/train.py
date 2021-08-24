import os
from keras.optimizers import Adam
import multiprocessing

from utils import preprocess_instances, BatchPreProcessor
from models import get_baseline_convolutional_encoder, build_siamese_net
from librispeech import LibriSpeechDataset

LIBRISPEECH_SAMPLING_RATE = 16000

##############
# Parameters #
##############
n_seconds = 3
downsampling = 4
batchsize = 64
filters = 64
embedding_dimension = 64
dropout = 0.0
training_set = ['train-clean-100', 'train-clean-360']
validation_set = 'dev-clean'
pad = False
num_epochs = 50
evaluate_every_n_batches = 500
num_evaluation_tasks = 500
n_shot_classification = 1
k_way_classification = 5

# Derived parameters
input_length = int(LIBRISPEECH_SAMPLING_RATE * n_seconds / downsampling)
param_str = 'siamese__filters_{}__embed_{}__drop_{}__pad={}'.format(filters, embedding_dimension, dropout, pad  )


###################
# Create datasets #
###################
train = LibriSpeechDataset(training_set, n_seconds, pad=pad)
valid = LibriSpeechDataset(validation_set, n_seconds, stochastic=False, pad=pad)

batch_preprocessor = BatchPreProcessor('siamese', preprocess_instances(downsampling))
train_generator = (batch_preprocessor(batch) for batch in train.yield_verification_batches(batchsize))
valid_generator = (batch_preprocessor(batch) for batch in valid.yield_verification_batches(batchsize))


################
# Define model #
################
encoder = get_baseline_convolutional_encoder(filters, embedding_dimension, dropout=dropout)
siamese = build_siamese_net(encoder, (input_length, 1), distance_metric='uniform_euclidean')
opt = Adam(clipnorm=1.)
siamese.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])


#################
# Training Loop #
#################
siamese.fit_generator(
    generator=train_generator,
    steps_per_epoch=evaluate_every_n_batches,
    validation_data=valid_generator,
    validation_steps=100,
    epochs=num_epochs,
    workers=multiprocessing.cpu_count(),
    use_multiprocessing=True,
)
