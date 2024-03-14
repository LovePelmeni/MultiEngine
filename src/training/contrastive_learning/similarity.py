from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
import librosa.feature
import numpy
from scipy.ndimage.filters import convolve
from scipy.signal import convolve2d

class SSIM(object):

    def gaussian(self, window_size, sigma):
        gauss = numpy.array([numpy.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
        return gauss / gauss.sum()

    def create_window(self, window_size, channel):
        _1D_window = self.gaussian(window_size, 1.5).reshape(-1)
        _2D_window = numpy.outer(_1D_window, _1D_window)
        window = numpy.stack([_2D_window, _2D_window, _2D_window], axis=-1)
        window = window * (channel == 1)
        return window

    def get_window_size(self, window_size, sigma):
        window = numpy.zeros((window_size, window_size))
        window[window_size//2, window_size//2] = 1
        return window * 0.25

    def _ssim(self, img1, img2, window, window_size, channel, size_average=True):
        mu1 = convolve(img1, window, mode='constant', cval=0)
        mu2 = convolve(img2, window, mode='constant', cval=0)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = convolve(img1 * img1, window, mode='constant', cval=0) - mu1_sq
        sigma2_sq = convolve(img2 * img2, window, mode='constant', cval=0) - mu2_sq
        sigma12 = convolve(img1 * img2, window, mode='constant', cval=0) - mu1_mu2

        C1 = (0.01 * 255) ** 2
        C2 = (0.03 * 255) ** 2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        if size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(1).mean(1).mean(1)

    def compute(self, 
        img1: numpy.ndarray, 
        img2: numpy.ndarray, 
        window_size=11, 
        window=None, 
        size_average=True,
        val_range=None
    ):
        """
        Calculates SSIM Score between a pair 
        of images to find similarity between them.
        """
        (_, channel, _, _) = img1.shape
        window = self.create_window(window_size, channel)
        
        if val_range is None:
            if numpy.max(img1) > 128:
                max_val = 255
            else:
                max_val = 1

            if numpy.min(img1) < -0.5:
                min_val = -1
            else:
                min_val = 0
            L = max_val - min_val
        else:
            L = val_range

        padd = 0
        (batch, channel, height, width) = img1.shape
        mu1 = convolve2d(img1, window, padding=padd, groups=channel)
        mu2 = convolve2d(img2, window, padding=padd, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = convolve2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
        sigma2_sq = convolve2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
        sigma12 = convolve2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2

        C1 = (0.01 * L) ** 2
        C2 = (0.03 * L) ** 2

        ssim_n = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
        ssim_d = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
        ssim = ssim_n / ssim_d

        if size_average:
            ssim = ssim.mean()
        else:
            ssim = ssim.mean(1).mean(1).mean(1)
        return ssim

def calculate_similarity(sentence1, sentence2):
    """
    Calculates similarity between 2 sentences
    by computing cosine similarity score of Word2Vec
    generated sentence embeddings
    """
    tokens1 = simple_preprocess(sentence1)
    tokens2 = simple_preprocess(sentence2)
    sentences = [tokens1, tokens2]
    model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, sg=0)
    vector1 = numpy.mean([model.wv[token] for token in tokens1], axis=0)
    vector2 = numpy.mean([model.wv[token] for token in tokens2], axis=0)
    similarity = numpy.dot(vector1, vector2) / (numpy.linalg.norm(vector1) * numpy.linalg.norm(vector2))
    return similarity 

def calculate_audio_similarity(signal1, signal2, sr1, sr2):
    """
    Calculates similarity between 2 audio files
    using cosine similarity of MFC Coefficients
    """
    # Extract MFCCs
    mfcc1 = librosa.feature.mfcc(signal1, sr1)
    mfcc2 = librosa.feature.mfcc(signal2, sr2)

    # Compute the mean MFCC for each audio file
    mean_mfcc1 = numpy.mean(mfcc1, axis=1)
    mean_mfcc2 = numpy.mean(mfcc2, axis=1)

    # Reshape to 2D arrays for cosine similarity
    mean_mfcc1 = mean_mfcc1.reshape(1, -1)
    mean_mfcc2 = mean_mfcc2.reshape(1, -1)
    cos_sim = numpy.dot(mean_mfcc1, mean_mfcc2) / (
    numpy.linalg.norm(mean_mfcc1) * numpy.linalg.norm(mean_mfcc2))
    return cos_sim
