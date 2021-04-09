import numpy as np


def NSF_DO(x, T):

	t1 = 14.59
	t2 = - 0.3955 * T
	t3 = 0.0072 * T**2
	t4 = - 0.0000619 * T**3

	x = 100 * x / (t1 + t2 + t3 + t4)

	if x < 40:
		return 0.18 + 0.66*x
	elif x < 100:
		return -13.55 + 1.17*x
	elif x < 140:
		return 163.34 - 0.62*x
	else:
		return 50


def NSF_FC(x):

	if x < 1:
		return 100
	elif x < 1000:
		return 97.2 - 26.6 * np.log10(x)
	elif x < 1000000:
		return 42.33 - 7.75 * np.log10(x)
	else:
		return 2


def NSF_pH(x):

	if x < 2 or x >= 12:
		return 0
	elif x < 5:
		return 16.1 + 7.35*x
	elif x < 7.3:
		return -142.67 + 33.5*x
	elif x < 10:
		return 316.96 - 29.85*x
	elif x < 12:
		return 96.17 - 8*x


def NSF_BOD(x):

	if x < 10:
		return 96.67 - 7*x
	elif x < 30:
		return 38.9 - 1.23*x
	else:
		return 2


def compute_wqi(DO, FC, pH, BOD, T):

	wqi = []
	for d, f, p, b, t in zip(DO, FC, pH, BOD, T):

		t1 = 0.31 * NSF_DO(d, t)
		t2 = 0.28 * NSF_FC(f)
		t3 = 0.22 * NSF_pH(p)
		t4 = 0.19 * NSF_BOD(b)

		wqi.append(t1+t2+t3+t4)

	return np.array(wqi)