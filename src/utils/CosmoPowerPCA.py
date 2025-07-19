import numpy as np
from sklearn.decomposition import IncrementalPCA
from sklearn.preprocessing import StandardScaler
import pickle


class CosmoPowerPCA:
    def __init__(self, n_pcas, features, modes):
        """
        PCA降维模块。
        参数:
            n_pcas: 主成分数
            features: 训练数据（log10OmegaGW，形状为 [n_samples, n_features]）
            modes: 频率f（形状为 [n_features]）
        """
        self.n_pcas = n_pcas
        self.modes = modes  # 频率f，256维
        self.feature_scaler = StandardScaler()
        self.pca = IncrementalPCA(n_components=n_pcas)

        # 标准化特征
        self.features_standard = self.feature_scaler.fit_transform(features)

        # 训练PCA
        self.pca.fit(self.features_standard)

        # 保存PCA参数
        self.pca_mean = self.feature_scaler.mean_
        self.pca_std = self.feature_scaler.scale_
        self.pca_components = self.pca.components_

        # 计算累积方差比例
        self.explained_variance_ratio = np.sum(self.pca.explained_variance_ratio_)
        print(f"PCA保留的方差比例: {self.explained_variance_ratio:.4f}")

    def transform(self, features):
        """将特征投影到PCA空间，生成系数"""
        features_standard = self.feature_scaler.transform(features)
        return self.pca.transform(features_standard)

    def inverse_transform(self, pca_coeffs):
        """从PCA系数重建特征"""
        features_standard = self.pca.inverse_transform(pca_coeffs)
        return self.feature_scaler.inverse_transform(features_standard)

    def save(self, filename):
        """保存PCA模型"""
        with open(filename, 'wb') as f:
            pickle.dump({
                'n_pcas': self.n_pcas,
                'modes': self.modes,
                'feature_scaler': self.feature_scaler,
                'pca': self.pca,
                'explained_variance_ratio': self.explained_variance_ratio
            }, f)

    @staticmethod
    def load(filename):
        """加载PCA模型"""
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        pca_obj = CosmoPowerPCA(data['n_pcas'], np.zeros((1, len(data['modes']))), data['modes'])
        pca_obj.feature_scaler = data['feature_scaler']
        pca_obj.pca = data['pca']
        pca_obj.explained_variance_ratio = data['explained_variance_ratio']
        return pca_obj