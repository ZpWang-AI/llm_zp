from utils_zp import *


class LLaMAFactoryBase:
    @property
    def dict(self) -> dict: 
        dic = dataclasses.asdict(self)
        for k in list(dic.keys()):
            if dic[k] is None: del dic[k]
            elif isinstance(dic[k], path): dic[k] = str(dic[k])
        return dic
    
    @property
    def dic(self): return self.dict

    @staticmethod
    def run_cmd(cmd, log_filepath, env_dict=None):
        env = os.environ.copy()
        if env_dict:
            for k in env_dict: env[k] = env_dict[k]
        
        try:
            print('>', ' '.join(cmd))
            print(f"> log file: {log_filepath}")
            with open(log_filepath, "w") as log_file:
                result = subprocess.run(
                    cmd,
                    env=env,
                    stdout=log_file,
                    # stderr=subprocess.PIPE,  # 分开捕获错误
                    stderr=log_file,
                    text=True,
                    check=False,  # 不自动抛出异常
                    encoding=utf8,
                )
            # if result.stderr:
            #     with open(log_filepath, "a", encoding=utf8) as log_file:
            #         log_file.write(f"\n{gap_line('ERROR')}\n")
            #         log_file.write(result.stderr)
            if result.returncode != 0:
                print(f"> Fail! return code: {result.returncode}")
                return False
            else:
                print(f'> Done!')
                return True
            
        except Exception as e:
            print(f"Error: {e}")
            return False


@dataclass
class LLaMAFactorySFTLora(LLaMAFactoryBase):
    model_name_or_path:str
    adapter_name_or_path:str
    template:str
    dataset:str
    output_dir:str
    
    ### model performance
    # ===========================================
    lora_rank:int = 8
    learning_rate:float = 1.0e-4
    num_train_epochs:float =  3.0
    max_samples:int = 1000000000
    # ===========================================

    ### CUDA memory
    # ===========================================
    # 128*128 = 16384
    # 256*256 = 65536
    # 768*768 = 589824
    image_max_pixels:int = 16384
    video_max_pixels:int = 16384
    video_fps:float = 2.0
    video_maxlen:int = 32
    # 'cutoff_len': 4096,  # qwenvl
    cutoff_len:int = 32768  # internvl
    # ===========================================

    trust_remote_code:bool = True

    ### method
    stage:str = 'sft'
    do_train:bool = True
    flash_attn:str = 'auto'
    finetuning_type:str = 'lora'
    lora_target:str = 'all'

    ### data
    overwrite_cache:bool = True
    preprocessing_num_workers:int = 16
    dataloader_num_workers:int = 4
    tokenized_path:str = None  # str(DATA_DIR_SFT / 'tokenized_dataset' / f'{dataset_name}__{template}')

    ### output
    logging_steps:int = 10
    save_steps:int = 500
    plot_loss:bool = True
    overwrite_output_dir:bool = True
    save_only_model:bool = False
    report_to:str = None  # choices: [none, wandb, tensorboard, swanlab, mlflow]

    ### train
    per_device_train_batch_size:int = 1
    gradient_accumulation_steps:int = 8
    lr_scheduler_type:str = 'cosine'
    warmup_ratio:float =  0.1
    bf16:bool = True
    ddp_timeout:int = 180000000
    resume_from_checkpoint:str = None

    def start(self, cuda_visible, llama_factory_dir):
        llama_factory_dir = path(llama_factory_dir)
        assert llama_factory_dir.exists()
        os.chdir(llama_factory_dir)
        
        output_dir = path(self.output_dir); make_path(output_dir)
        time_str = Datetime_().format_str('%Y-%m-%d_%H-%M-%S')
        yaml_file = output_dir / f'sftlora_{time_str}.yaml'
        auto_dump(self.dict, yaml_file)
        log_filepath = output_dir / 'log.txt'
        cmd = ["llamafactory-cli", "train", str(yaml_file)]
        
        self.run_cmd(
            cmd=cmd, 
            log_filepath=log_filepath, 
            env_dict={"CUDA_VISIBLE_DEVICES": cuda_visible},
        )


@dataclass
class LLaMAFactoryMergeLora(LLaMAFactoryBase):
    model_name_or_path:str
    adapter_name_or_path:str
    template:str
    export_dir:str

    trust_remote_code:bool = True
    ### export
    export_size:int = 5
    export_device:str = 'auto' # choices: [cpu, auto]
    export_legacy_format:str = 'false'

    def start(self,):
        output_dir = path(self.export_dir); make_path(output_dir)
        time_str = Datetime_().format_str('%Y-%m-%d_%H-%M-%S')
        yaml_file = output_dir / f'mergelora_{time_str}.yaml'
        auto_dump(self.dict, yaml_file)
        log_filepath = output_dir / 'log.txt'
        cmd = ["llamafactory-cli", "export", str(yaml_file)]

        self.run_cmd(
            cmd=cmd, log_filepath=log_filepath,
        )
