# Copied from: https://github.com/IST-DASLab/gptq
import math
import time

import torch
import torch.nn as nn
import transformers


DEBUG = False 

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


class GPTQ:
	def __init__(self, layer):
		self.layer = layer
		self.dev = self.layer.weight.device
		W = layer.weight.data.clone()
		if isinstance(self.layer, nn.Conv2d):
			W = W.flatten(1)
		if isinstance(self.layer, transformers.Conv1D):
			W = W.t()
		self.rows = W.shape[0]
		self.columns = W.shape[1]
		self.H = torch.zeros((self.columns, self.columns), device=self.dev)
		self.nsamples = 0

	def add_batch(self, inp, out):
		if DEBUG:
			self.inp1 = inp
			self.out1 = out
		if len(inp.shape) == 2:
			inp = inp.unsqueeze(0)
		tmp = inp.shape[0]
		if isinstance(self.layer, nn.Linear) or isinstance(self.layer, transformers.Conv1D):
			if len(inp.shape) == 3:
				inp = inp.reshape((-1, inp.shape[-1]))
			inp = inp.t()
		if isinstance(self.layer, nn.Conv2d):
			unfold = nn.Unfold(
				self.layer.kernel_size,
				dilation=self.layer.dilation,
				padding=self.layer.padding,
				stride=self.layer.stride
			)
			inp = unfold(inp)
			inp = inp.permute([1, 0, 2])
			inp = inp.flatten(1)
		self.H *= self.nsamples / (self.nsamples + tmp)
		self.nsamples += tmp
		# inp = inp.float()
		inp = math.sqrt(2 / self.nsamples) * inp.float()
		# self.H += 2 / self.nsamples * inp.matmul(inp.t())
		self.H += inp.matmul(inp.t())

	def fasterquant(
		self, blocksize=128, percdamp=.01, groupsize=-1, actorder=False
	):
		W = self.layer.weight.data.clone()
		if isinstance(self.layer, nn.Conv2d):
			W = W.flatten(1)
		if isinstance(self.layer, transformers.Conv1D):
			W = W.t()
		W = W.float()

		tick = time.time()

		if not self.quantizer.ready():
			self.quantizer.find_params(W, weight=True)

		H = self.H
		del self.H
		dead = torch.diag(H) == 0
		H[dead, dead] = 1
		W[:, dead] = 0

		if actorder:
			perm = torch.argsort(torch.diag(H), descending=True)
			W = W[:, perm]
			H = H[perm][:, perm]

		Losses = torch.zeros_like(W)
		Q = torch.zeros_like(W)

		damp = percdamp * torch.mean(torch.diag(H))
		diag = torch.arange(self.columns, device=self.dev)
		H[diag, diag] += damp
		H = torch.linalg.cholesky(H)
		H = torch.cholesky_inverse(H)
		H = torch.linalg.cholesky(H, upper=True)
		Hinv = H
		
		scale = []
		zero = []
		now_idx = 1

		for i1 in range(0, self.columns, blocksize):
			i2 = min(i1 + blocksize, self.columns)
			count = i2 - i1

			W1 = W[:, i1:i2].clone()
			Q1 = torch.zeros_like(W1)
			Err1 = torch.zeros_like(W1)
			Losses1 = torch.zeros_like(W1)
			Hinv1 = Hinv[i1:i2, i1:i2]

			for i in range(count):
				w = W1[:, i]
				d = Hinv1[i, i]

				if groupsize != -1:
					if (i1 + i) % groupsize == 0:
						self.quantizer.find_params(W[:, (i1 + i):(i1 + i + groupsize)], weight=True)
					
					if ((i1 + i) // groupsize) - now_idx == -1:
						scale.append(self.quantizer.scale)
						zero.append(self.quantizer.zero)
						now_idx += 1

				q = quantize(
					w.unsqueeze(1), self.quantizer.scale, self.quantizer.zero, self.quantizer.maxq
				).flatten()
				Q1[:, i] = q
				Losses1[:, i] = (w - q) ** 2 / d ** 2

				err1 = (w - q) / d
				W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
				Err1[:, i] = err1

			Q[:, i1:i2] = Q1
			Losses[:, i1:i2] = Losses1 / 2

			W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

			if DEBUG:
				self.layer.weight.data[:, :i2] = Q[:, :i2]
				self.layer.weight.data[:, i2:] = W[:, i2:]
				print(torch.sum((self.layer(self.inp1) - self.out1) ** 2))
				print(torch.sum(Losses))

		torch.cuda.synchronize()
		print('time %.2f' % (time.time() - tick))
		print('error', torch.sum(Losses).item())
		
		if actorder:
			invperm = torch.argsort(perm)
			Q = Q[:, invperm]

		if isinstance(self.layer, transformers.Conv1D):
			Q = Q.t()
		self.layer.weight.data = Q.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)
		if DEBUG:
			print(torch.sum((self.layer(self.inp1) - self.out1) ** 2))
			
		if scale == []:
			scale.append(self.quantizer.scale)
			zero.append(self.quantizer.zero)
		scale = torch.cat(scale,dim=1)
		zero = torch.cat(zero,dim=1)
		return scale,zero
			
	def free(self):
		if DEBUG:
			self.inp1 = None
			self.out1 = None
		self.H = None
		self.Losses = None
		self.Trace = None
		torch.cuda.empty_cache()


def quantize(x, scale, zero, maxq):
	if maxq < 0:
		return (x > scale / 2).float() * scale + (x < zero / 2).float() * zero
	q = torch.clamp(torch.round(x / scale) + zero, 0, maxq)
	return scale * (q - zero)


class Quantizer(nn.Module):
	def __init__(self, shape=1):
		super(Quantizer, self).__init__()
		self.register_buffer('maxq', torch.tensor(0))
		self.register_buffer('scale', torch.zeros(shape))
		self.register_buffer('zero', torch.zeros(shape))

	def configure(
		self,
		bits, perchannel=False, sym=True, 
		mse=False, norm=2.4, grid=100, maxshrink=.8,
		trits=False
		):
		
		self.maxq = torch.tensor(2 ** bits - 1)
		self.perchannel = perchannel
		self.sym = sym
		self.mse = mse
		self.norm = norm
		self.grid = grid
		self.maxshrink = maxshrink 
		if trits:
			self.maxq = torch.tensor(-1) 

	def find_params(self, x, weight=False):
		dev = x.device
		self.maxq = self.maxq.to(dev)

		shape = x.shape
		if self.perchannel:
			if weight:
				x = x.flatten(1)
			else:
				if len(shape) == 4:
					x = x.permute([1, 0, 2, 3])
					x = x.flatten(1)
				if len(shape) == 3:
					x = x.reshape((-1, shape[-1])).t()
				if len(shape) == 2:
					x = x.t()
		else:
			x = x.flatten().unsqueeze(0)

		tmp = torch.zeros(x.shape[0], device=dev)
		xmin = torch.minimum(x.min(1)[0], tmp)
		xmax = torch.maximum(x.max(1)[0], tmp)

		if self.sym:
			xmax = torch.maximum(torch.abs(xmin), xmax)
			tmp = xmin < 0
			if torch.any(tmp):
				xmin[tmp] = -xmax[tmp]
		tmp = (xmin == 0) & (xmax == 0)
		xmin[tmp] = -1
		xmax[tmp] = +1

		if self.maxq < 0:
			self.scale = xmax
			self.zero = xmin
		else:
			self.scale = (xmax - xmin) / self.maxq
			if self.sym:
				self.zero = torch.full_like(self.scale, (self.maxq + 1) / 2)
			else:
				self.zero = torch.round(-xmin / self.scale)

		if self.mse:
			best = torch.full([x.shape[0]], float('inf'), device=dev)
			for i in range(int(self.maxshrink * self.grid)):
				p = 1 - i / self.grid 
				xmin1 = p * xmin
				xmax1 = p * xmax
				scale1 = (xmax1 - xmin1) / self.maxq
				zero1 = torch.round(-xmin1 / scale1) if not self.sym else self.zero
				q = quantize(x, scale1.unsqueeze(1), zero1.unsqueeze(1), self.maxq)
				q -= x
				q.abs_()
				q.pow_(self.norm)
				err = torch.sum(q, 1)
				tmp = err < best
				if torch.any(tmp):
					best[tmp] = err[tmp]
					self.scale[tmp] = scale1[tmp]
					self.zero[tmp] = zero1[tmp]
		if not self.perchannel:
			if weight:
				tmp = shape[0]
			else:
				tmp = shape[1] if len(shape) != 3 else shape[2]
			self.scale = self.scale.repeat(tmp)
			self.zero = self.zero.repeat(tmp)

		if weight:
			shape = [-1] + [1] * (len(shape) - 1)
			self.scale = self.scale.reshape(shape)
			self.zero = self.zero.reshape(shape)
			return
		if len(shape) == 4:
			self.scale = self.scale.reshape((1, -1, 1, 1))
			self.zero = self.zero.reshape((1, -1, 1, 1))
		if len(shape) == 3:
			self.scale = self.scale.reshape((1, 1, -1))
			self.zero = self.zero.reshape((1, 1, -1)) 
		if len(shape) == 2:
			self.scale = self.scale.unsqueeze(0)
			self.zero = self.zero.unsqueeze(0)

	def quantize(self, x):
		if self.ready():
			return quantize(x, self.scale, self.zero, self.maxq)
		return x

	def enabled(self):
		return self.maxq > 0

	def ready(self):
		return torch.all(self.scale != 0)