import numpy as np 
import matplotlib.pyplot as plt
from datetime import datetime

def nowstring():
	"""Generate a unite datetime for saving data
	"""
	return datetime.now().strftime("%Y-%m-%d_%H-%M-%S") 


def save_fig(loc):
	plt.savefig(str(loc)+'.pdf',dpi=300, bbox_inches = 'tight',
    pad_inches = 0)
	plt.savefig(str(loc)+'.png',dpi=300, bbox_inches = 'tight',
	    pad_inches = 0)
	plt.savefig(str(loc)+'.svg',dpi=300, bbox_inches = 'tight',
	    pad_inches = 0)
	plt.savefig(str(loc)+'.png',dpi=300, bbox_inches = 'tight',
	    pad_inches = 0)
	plt.savefig(str(loc)+'.jpg',dpi=150, bbox_inches = 'tight',
	    pad_inches = 0)