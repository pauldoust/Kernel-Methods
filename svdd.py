""" Supported Kernels: Linear, Polynomial, Gaussian (RBF)

For one class case - use decision_rule() method for further plotting (Gives smoother plot).
To predict labels/outliers in +-1, use predict().
"""

import time
import math
import numpy as np
import quadprog as qd

class SVDD:
	def __init__(self, C = 0.5, gamma = 3., kernel = 'rbf', degree = 3):
		"""
		1. C is the parameter for tradeoff between error and fitting data, allows for outliers. 
		C>=1 is no slack case.
		2. gamma is for rbf kernel
		3. degree and coeff for polynomial
		"""
		self.C = C
		self.gamma = gamma
		self.kernel = kernel
		if kernel == None or kernel == 'linear':
			self.degree = 1 # If kernel is linear, degree is set to 1
		else:
			self.degree = degree
		#if kernel == 'linear':
		#	self.coeff = 0 # If kernel is linear, coeff is set to 0
		#else:
		#	self.coeff = coeff
		
	def fit(self, X, y, sample_weight = None):
		"""
		X: nd array (Input)
		y: n array (Targets = +1, outliers = -1)
		If no y is given, y = array of 1s. All are target.
		"""
		if y is None:
			y = np.ones(X.shape[0], dtype = 'int')
		else:
			y = y

		G, q, P, h, A, b = self.op(X, y)
		alpha = y*self.solve_op(self.findnearestPSD(G), q, P, h, A, b)
		self.support_vectors = X[abs(alpha) > 1e-6,:] # Get support vectors
		# Choose any very small quantity >0 for gammas.
		self.support = np.squeeze(np.where(abs(alpha) > 1e-6)) 
		self.alpha_coef = alpha[abs(alpha) > 1e-6] # FInds alphas satisfying inequality
		
		# Get radius of support vectors and get threshold (Set to mean here.)
		# Different threshold can be selected.
		R2, _ = self.findradius(self.support_vectors) # Get R2 for support vectors
		self.threshold = np.mean(R2) # Set threshold
		
		return self
	
	def predict(self, X):
		"""
		Predict labels. Use for classification (Two Class case)
		For outlier detection: Use decision_rule
		"""
		y = np.sign(self.decision_rule(X)).astype('int')
		y[y == 0] = 1 
		return y
	
	def decision_rule(self, X):
		"""
		Define decision rule to classify/find if they are outlier
		"""
		radius, _ = self.findradius(X)
		return self.threshold - radius
		
	def findradius(self, X):
		"""
		Find R2, uses R2 formula
		"""
		k = 0 # Initial hyper parameters
		l = 0
		m = 0
		
		for i in range(len(self.alpha_coef)):
			K_xz, dK_xz = self.find_kernel(self.support_vectors[i,:], X)
			k = k + self.alpha_coef[i]*K_xz
			m = m + self.alpha_coef[i]*dK_xz
			for j in range(len(self.alpha_coef)): # Third term
				K_xx, _ = self.find_kernel(self.support_vectors[i,:], self.support_vectors[j,:])
				l = l + self.alpha_coef[i]*self.alpha_coef[j]*K_xx
				
		R2 = 1 - 2*k + l
		dR2 = -2*m
		
		return R2, dR2
		
	def find_kernel(self, x, z):
		"""
		rbf and polynomial kernels, and their derivatives
		"""

		if z.ndim > 1:
			K = np.exp(-self.gamma*np.linalg.norm(x - z, axis = 1)**2)
			dK = np.dot(np.diag(np.exp(-self.gamma*np.linalg.norm(x - z, axis = 1)**2)), 2*self.gamma*(x - z))
		else:
			K = np.exp(-self.gamma*np.linalg.norm(x - z)**2)
			dK = 2*self.gamma*(x - z)*np.exp(-self.gamma*np.linalg.norm(x - z)**2)
		#elif self.kernel == 'poly':
		#	K = np.dot(x,z)**self.degree
		#	dK = 2*self.degree*np.dot(x,z)**(self.degree-1)
		# Got UnboundLocalError when created polynomial here 

		return K, dK
	def polynomial_kernel(self, x, z):
		K = np.dot((x-z).T, (x-z))**self.degree
		dK = 2*(self.degree)*(np.dot((x-z).T,(x-z))**(self.degree-1))*(x-z)

		return K, dK

	def op(self, X, y):
		"""
		Build optimization problem to solve with optimization libraries.
		"""
		# G: Gram Matrix and its derivative
		G = np.eye(len(y))
		for i in range(len(y) - 1):
			for j in range(i + 1, len(y)):
				if self.kernel == None:
					K_xx, _ = self.polynomial_kernel(X[i,:],X[j,:])
				if self.kernel == 'rbf':
					K_xx, _ = self.find_kernel(X[i,:],X[j,:])
				if self.kernel == 'polynomial':
					K_xx, _ = self.polynomial_kernel(X[i,:],X[j,:])
				G[i,j] = y[i]*y[j]*K_xx
		G = self.findnearestPSD(G + G.T - np.eye(len(y)))
		q = np.zeros(len(y)) # Array of zeros
		P = np.vstack((-np.eye(len(y)),np.eye(len(y)))) 
		h = np.hstack((np.zeros(len(y)),self.C*np.ones(len(y))))
		A = y
		b = 1.	
		return G, q, P, h, A, b
	
	def solve_op(self, G, q, P = None, h = None, A = None, b = None):
		"""
		Accepts op: optimization problem
		Returns alphas
		"""
		qp_G = .5 * (G + G.T)  
		qp_a = -q
		if A is not None:
			qp_C = -np.vstack([A, P]).T
			qp_b = -np.hstack([b, h])
			meq = 1#A.shape[0]
		else: 
			qp_C = -G.T
			qp_b = -h
			meq = 0

		return qd.solve_qp(qp_G, qp_a, qp_C, qp_b, meq)[0]
	
	def findnearestPSD(self, A):
		"""
		Finds the nearest PSD matrix. quadprog needs it to be such for covergence.
		"""
	
		B = (A + A.T) / 2
		_, s, V = np.linalg.svd(B)
	
		H = np.dot(V.T, np.dot(np.diag(s), V))
	
		A2 = (B + H) / 2
	
		A3 = (A2 + A2.T) / 2
	
		if self.findifPSD(A3):
			return A3
	
		spacing = np.spacing(np.linalg.norm(A))
		I = np.eye(A.shape[0])
		k = 1
		while not self.findifPSD(A3):
			mineig = np.min(np.real(np.linalg.eigvals(A3)))
			A3 += I * (-mineig * k**2 + spacing)
			k += 1
	
		return A3
	
	def findifPSD(self, B): # Check if matrix is positive semidefinite.
		try:
			_ = np.linalg.cholesky(B)
			return True
		except np.linalg.LinAlgError:
			return False