from os import listdir
from os.path import isfile, join

import soundfile as sf
import numpy as np
import random


class ReadData:
    def __init__(self, batchAudioLength=1., batchSize=100, fileNumperSpeaker=2, downsampling=4):
        self.batchAudioLength = batchAudioLength
        self.batchSize = batchSize
        self.fileNumperSpeaker = fileNumperSpeaker
        self.downsampling = downsampling

        self.frame_length = 0.025
        self.frame_stride = 0.010
        self.sampleRate = 16000

        self.audioFilePath = '../LibriSpeech/train-clean-360/'
        self.speakerList = [f for f in listdir(self.audioFilePath) if f[0] != '.']

        print('The number of speaker: {}'.format(len(self.speakerList)))

        self.fileLists = []
        for speaker in self.speakerList:
            speakerDir = join(self.audioFilePath, speaker)
            speakerDirList = [join(speakerDir, f) for f in listdir(speakerDir) if f[0] != '.']

            totalFileList = []
            for chapter in speakerDirList:
                fileList = [join(chapter, f) for f in listdir(chapter) 
                    if isfile(join(chapter, f)) and ('.flac' in f)]
                totalFileList += fileList
            
            self.fileLists.append(totalFileList)
        
        #Test
        #self.speakerList = self.speakerList[80:100]
    

    def nextBatch(self):
        batch_data_x_1 = []
        batch_data_x_2 = []
        batch_data_y = []

        dataNum = 0
        while True:
            # same speaker
            if dataNum < self.batchSize/2:
                speaker1 = speaker2 = random.randint(0, len(self.speakerList)-1)
            # different speaker
            else:
                speaker1 = random.randint(0, len(self.speakerList)-1)
                possibleNumbers = list(range(0, speaker1)) + list(range(speaker1+1, len(self.speakerList)))
                speaker2 = random.choice(possibleNumbers)
            

            # Read data from disk. 
            readFileNumber = random.randint(0, len(self.fileLists[speaker1])-1)
            data_1, self.sampleRate = sf.read(self.fileLists[speaker1][readFileNumber])

            readFileNumber = random.randint(0, len(self.fileLists[speaker2])-1)
            data_2, self.sampleRate = sf.read(self.fileLists[speaker2][readFileNumber])
            
            # Check if audio length is longer than batch audio length. 
            audioLength = min(len(data_1), len(data_2)) / self.sampleRate
            
            if audioLength > self.batchAudioLength:
                startPoint = random.randint(0, 
                    int(len(data_1)/self.sampleRate-self.batchAudioLength))
                batch_data_x_1.append(
                    data_1[int(startPoint*self.sampleRate):int((startPoint+self.batchAudioLength)*self.sampleRate)])
                
                startPoint = random.randint(0, 
                    int(len(data_2)/self.sampleRate-self.batchAudioLength))
                batch_data_x_2.append(
                    data_2[int(startPoint*self.sampleRate):int((startPoint+self.batchAudioLength)*self.sampleRate)])
                
                #batch_data_y.append([0.] if speaker1 == speaker2 else [1.])
                dataNum += 1
            
            if len(batch_data_x_1) >= self.batchSize:
                #shuffleList = list(range(self.batchSize))
                #random.shuffle(shuffleList)

                ret_batch_data_x_1 = self.Whitening(self.Downsampling(np.array(batch_data_x_1), self.downsampling))
                ret_batch_data_x_2 = self.Whitening(self.Downsampling(np.array(batch_data_x_2), self.downsampling))

                #ret_batch_data_y = np.array(batch_data_y)
                ret_batch_data_y = np.append(np.zeros(self.batchSize/2), np.ones(self.batchSize/2))[:, np.newaxis]

                return ret_batch_data_x_1, ret_batch_data_x_2, ret_batch_data_y


    def Downsampling(self, batch, downsampling=4):
        batch = batch[:, ::downsampling]
        
        return batch
    
    
    def Whitening(self, batch, rms=0.038021):
        sample_wise_mean = batch.mean(axis=1)
        whitend_batch = batch - np.tile(sample_wise_mean, (batch.shape[1], 1)).transpose((1, 0))

        sample_wise_rescaling = rms / np.sqrt(np.power(batch, 2).mean())
        whitend_batch = whitend_batch * np.tile(sample_wise_rescaling, (batch.shape[1], 1)).transpose((1, 0))

        return whitend_batch
    

    def NextBatchTest(self):
        batch_data_x_1 = np.random.rand(self.batchSize, int(self.sampleRate*self.batchAudioLength/self.downsampling))
        batch_data_x_2 = np.random.rand(self.batchSize, int(self.sampleRate*self.batchAudioLength/self.downsampling))
        batch_data_y = np.random.randint(0, 2, size=(self.batchSize, 1))

        return batch_data_x_1, batch_data_x_2, batch_data_y
    

### Test ###

#length = 3.
#batchSize=12
#
#readData = ReadData(batchAudioLength=length, batchSize=batchSize)
#for _ in range(10):
#    a, b, c = readData.nextBatch()
#    if len(a) != batchSize or len(b) != batchSize or len(c) != batchSize:
#        print('{} {} {}'.format(len(a), len(b), len(c)))
#
#print('Done!')

readData = ReadData(batchAudioLength=1., batchSize=100, fileNumperSpeaker=2)
a, b, c = readData.nextBatch()
#print(a.shape)
#print(c.shape)
#print(c)

#print(a[0])
#print()
#print(b[0])
#print()
#print(c)
#print(c.shape)
