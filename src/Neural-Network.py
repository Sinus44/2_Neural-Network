import random as rnd

class Neuron():
	def __init__(self):
		self.sum = 0
		self.out = 0
		self.delta = 0
		self.isBias = False

class NeuralNetwork:
	def __init__(self, inputCount, hidenLayers, outputs, learnRate=0.3, moment=0, useBias=False):
		self.learnRate = learnRate
		self.moment = moment
		self.useBias = useBias

		# Инициализация входного слоя
		inputs = [Neuron() for i in range(inputCount + (1 if self.useBias else 0))]

		if self.useBias:
			inputs[-1].isBias = True
			inputs[-1].out = 1 

		# Инициализация внутрених слоев
		self.hidenLayers = []
		for i in range(len(hidenLayers)):
			self.hidenLayers.append([])
			for j in range(hidenLayers[i]+(1 if self.useBias else 0)):
				self.hidenLayers[i].append(Neuron())

			if self.useBias:
				self.hidenLayers[i][-1].isBias = True
				self.hidenLayers[i][-1].out = 1 
		
		self.multiArr = [inputs] + self.hidenLayers + [[Neuron() for _ in range(outputs)]]
		
		# Инициализация весов
		self.w = []
		for i in range(len(self.multiArr) - 1):
			self.w.append([])
			for j in range(len(self.multiArr[i])):
				self.w[i].append([])
				for k in range(len(self.multiArr[i+1])):
					self.w[i][j].append(rnd.random())

		self.prevW = []
		for i in range(len(self.multiArr) - 1):
			self.prevW.append([])
			for j in range(len(self.multiArr[i])):
				self.prevW[i].append([])
				for k in range(len(self.multiArr[i+1])):
					self.prevW[i][j].append(0)
	
	def predict(self, inputs):
		# Вставка входных данных
		for neurons, inp in zip(self.multiArr[0], inputs):
			neurons.out = inp if not neurons.isBias else 1

		for i, layer in enumerate(self.multiArr[1:]):
			for j, neuron in enumerate(layer):
				neuron.sum = sum([lastNeuron.out * self.w[i][k][j] for k, lastNeuron in enumerate(self.multiArr[i])])
				neuron.out = self.activation(neuron.sum)

		return [neuron.out for neuron in self.multiArr[-1]]	
	
	def learn(self, data, out):
		res = self.predict(data)

		for i, mi in enumerate(self.multiArr[-1]):
			er = res[i] - out[i]
			mi.delta = er * self.derivativeActivation(res[i]) # delta for last layer

		# Для каждого промежутка
		for i in range(len(self.w) - 1, -1, -1):
			if i != len(self.w) - 1:
				# для каждого нейрона в левом слое
				for j, neuron in enumerate(self.multiArr[i+1]):
					neuron.delta = sum([self.w[i+1][j][k] * mik.delta for k, mik in enumerate(self.multiArr[i+2])]) * self.derivativeActivation(neuron.out)
			
			# Для каждого левого слоя
			for j, wij in enumerate(self.w[i]):
				# Для каждого правого слоя
				for k, wijk in enumerate(wij):

					deltaW = self.learnRate * self.multiArr[i][j].out * self.multiArr[i+1][k].delta
					self.w[i][j][k] -= deltaW
					if self.moment:
						self.w[i][j][k] -= self.prevW[i][j][k] * self.moment
						self.prevW[i][j][k] = deltaW

	def activation(self, x):
		return 1 / (1 + (2.71828182 ** (-x)))

	def derivativeActivation(self, x):
		return x * (1 - x)