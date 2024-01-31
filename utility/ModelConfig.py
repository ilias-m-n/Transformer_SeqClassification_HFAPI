class ModelConfig:

    def __init__(self, 
                 timestamp, 
                 base_model, 
                 task, 
                 loss_fct, 

                 from_hub,
                 dataset_name_hub,
                 dataset_name_local, 
                 path_dataset_local, 
                 
                 num_labels,
                 weight_scheme, 
                 class_weights,
                 eval_metrics,  
                 metric_best_model, 
                 
                 no_trials, 
                 frozen,
                 path_initial_training,
                 best_run):

        # set during initialization 
        self.timestamp_initial = timestamp
        self.base_model = base_model
        self.task = task
        self.loss_fct = loss_fct

        self.from_hub = from_hub
        self.dataset_name_hub = dataset_name_hub
        self.dataset_name_local = dataset_name_local
        self.path_dataset_local = path_dataset_local


        
        self.num_labels = num_labels
        self.weight_scheme = weight_scheme
        self.class_weights = class_weights
        self.eval_metrics = eval_metrics 
        self.metric_best_model = metric_best_model
        
        self.no_trails = no_trials
        self.frozen = frozen
        self.best_run = best_run
        self.path_initial_training = path_initial_training

        # set during final training after hyperparameter search
        self.timestamp_final = None
        self.path_final_training = None
        self.path_trained_model = None
        


    def __str__(self):
        output = f"""
        Model Config Information:

        Data: {self.timestamp}
        Base Model: {self.base_model}
        Task: {self.task}
        Dataset: {self.dataset}
        """

        return output