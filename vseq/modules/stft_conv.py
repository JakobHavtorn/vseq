
import torch
import torch.nn as nn
import numpy as np

class STFTConv(nn.Conv1d):

	def __init__(self, window_size=0.02, stride=0.01, sample_rate=16000, window_func=np.hanning):
		stride = int(sample_rate * stride)
		kernel_size = int(sample_rate * window_size)
		mirror = kernel_size // 2 + 1
		out_channels = mirror * 2
		
		super(STFTConv, self).__init__(in_channels=1,
									   out_channels=out_channels,
									   kernel_size=kernel_size,
									   stride=stride,
									   bias=False)

		window = window_func(kernel_size)
		fb = self.create_fourier_basis(kernel_size)
		fb_real = np.expand_dims(fb.real[:,:mirror], axis=1)
		fb_imag = np.expand_dims(fb.imag[:,:mirror], axis=1)
		fb_real = (fb_real.T * window).T
		fb_imag = (fb_imag.T * window).T
		stft_weight = np.concatenate([fb_real, fb_imag], axis=2).swapaxes(0, 2)

		with torch.no_grad():
			init_value = torch.FloatTensor(stft_weight)
			self.weight = nn.Parameter(data=init_value, requires_grad=True)

		# @TODO: Change to causal padding
		#self.pre_pad = nn.ConstantPad1d(padding=kernel_size // 2, value=0)

	@staticmethod
	def create_fourier_basis(N):
		n = np.tile(np.arange(N), (N, 1))
		nk = n * n.T
		return np.e ** (-1j * (2 * np.pi)/N * nk)

	def forward(self, x):
		#x = self.pre_pad(x)
		x = super(STFTConv, self).forward(x)
		x = torch.mul(x, x)
		m = self.out_channels // 2
		x = torch.sqrt(x[:, :m] + x[:, m:])
		x = x.unsqueeze(1)
		return x