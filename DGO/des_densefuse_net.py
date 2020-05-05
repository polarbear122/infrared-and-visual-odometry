# DenseFuse Network
# Encoder -> Addition/L1-norm -> Decoder

import tensorflow as tf

from des_encoder import Encoder
from des_decoder import Decoder
from des_fusion_addition import Strategy


class DenseFuseNet(object):
    def __init__(self, model_pre_path):
        self.encoder = Encoder(model_pre_path)
        self.decoder = Decoder(model_pre_path)

    def transform_addition(self, img1, img2):
        # encode image
        enc_1 = self.encoder.encode(img1)
        print('enc1:{}'.format(enc_1))
        enc_2 = self.encoder.encode(img2)
        print('enc2:{}'.format(enc_2))
        target_features = Strategy(enc_1, enc_2)
        print('target_features:{}'.format(target_features))
        # target_features = enc_c
        self.target_features = target_features
        # print('target_features:', target_features.shape)
        # print('enc1==enc2:{},enc1==target_feature:{}'.format(enc_1==enc_2,enc_1==target_features))
        # decode target features back to image
        generated_img = self.decoder.decode(target_features)
        print('generated_img:{}'.format(generated_img))
        return generated_img

    def transform_recons(self, img):
        # encode image
        enc = self.encoder.encode(img)
        target_features = enc
        self.target_features = target_features
        generated_img = self.decoder.decode(target_features)
        return generated_img


    def transform_encoder(self, img):
        # encode image
        enc = self.encoder.encode(img)
        return enc

    def transform_decoder(self, feature):
        # decode image
        generated_img = self.decoder.decode(feature)
        return generated_img

