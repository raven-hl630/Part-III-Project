multiprocessing folder:
	simulation.py:
		variable configuration
		PDF, CDF
		data generation
		cost function
		data fitting
		write configuration
		write fit result
	analysis.py:
		read configuration
		read fit result
		statistics
		generate plots

single folder:
	single.py:
		fixed configuration
		single pseudo-experiment only

fisher_information folder:
	fisher_information_integration.py:
		compute per-event Fisher information in S_f
		write integration result
	fisher_information_analysis.py:
		read integration result
		read simulation result
		statistics
		generate plots