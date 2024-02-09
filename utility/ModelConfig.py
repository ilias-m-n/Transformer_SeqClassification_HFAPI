class ModelConfig:

    def __init__(self, 
                 timestamp, 
                 base_model,
                 reset_model_head,
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
                 metric_direction,
                 
                 no_trials, 
                 frozen,
                 path_initial_training,
                 best_run,
                 hps_log_df,
                 flag_mv = False,
                 study_name = "",
                 path_study_db = ""):

        # set during initialization 
        self.timestamp_initial = timestamp
        self.base_model = base_model
        self.reset_model_head = reset_model_head
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
        self.metric_direction = metric_direction
        
        self.no_trials = no_trials
        self.frozen = frozen
        self.best_run = best_run
        self.path_initial_training = path_initial_training
        
        self.flag_mv = flag_mv

        # hyperparameter search
        self.hps_log_df = hps_log_df
        self.study_name = study_name
        self.path_study_db = path_study_db
        
        # set during final training after hyperparameter search
        self.timestamp_final = None
        self.path_final_training = None
        self.path_trained_model = None

        # set during performance evaluation
        self.training_log_df = None
        self.predictions = None
        self.evaluation_results = None
        self.confusion_matrix = None
        # majority voting
        self.evaluation_results_mv = None
        self.confusion_matrix_mv = None


    def __str__(self):
        output = f"""
        Model Config Information:

        Data: {self.timestamp}
        Base Model: {self.base_model}
        Task: {self.task}
        Dataset: {self.dataset}
        """

        return output