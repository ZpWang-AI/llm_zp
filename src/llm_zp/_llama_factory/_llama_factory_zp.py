from utils_zp import *


def LLaMAFactory_add_dataset(
    dataset_name:str, llamafactory_data_dir:Union[str,path],
    queries:List[str], answers:List[str], 
    overwrite_dataset:bool=False,
    images:List[List[str]]=None, videos:List[List[str]]=None,
):
    n = len(queries)
    assert len(answers)==n
    all_data = []
    columns = ['messages']
    for q,a in zip(queries,answers):
        all_data.append({
            'messages': [
                {'content':q, 'role':'user'},
                {'content':a, 'role':'assistant'},
            ]
        })
    if images is not None:
        assert len(images)==n
        columns.append('images')
        for _data,_images in zip(all_data,images):
            _data['images'] = _images
    if videos is not None:
        assert len(videos)==n
        columns.append('videos')
        for _data,_videos in zip(all_data,videos):
            _data['videos'] = _videos
    
    llamafactory_data_dir = path(llamafactory_data_dir)
    dataset_info_json = llamafactory_data_dir/'dataset_info.json'
    data_filename = llamafactory_data_dir/'dataset'/f'{dataset_name}.json'
    auto_dump(all_data, data_filename)
    _dataset_info = {
        "file_name": data_filename,
        "formatting": "sharegpt",
        "columns": {col:col for col in columns},
        "tags": {
            "role_tag": "role",
            "content_tag": "content",
            "user_tag": "user",
            "assistant_tag": "assistant",
            "system_tag": "system"
        }
    }
    all_dataset_info = auto_load(dataset_info_json) if dataset_info_json.exists() else {}
    if dataset_name in all_dataset_info:
        if not overwrite_dataset: raise Exception(f'{dataset_name} already exists')
    all_dataset_info[dataset_name] = _dataset_info
    auto_dump(all_dataset_info, dataset_info_json)
    print(f'> add {len(all_data)} samples into dataset {dataset_name}')


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
    def run_cmd(cmd, log_filepath=None, env_dict=None):
        env = os.environ.copy()
        if env_dict:
            for k in env_dict: env[k] = env_dict[k]
        
        print('>', ' '.join(cmd))
        
        if log_filepath is not None:
            print(f"> log file: {log_filepath}")

            def read_output(pipe, file_handle, is_stderr=False):
                try:
                    with pipe:
                        for line in iter(pipe.readline, ''):
                            if line:
                                file_handle.write(line)
                                file_handle.flush()
                                
                                if is_stderr:
                                    # sys.stderr.write(f"\033[91m{line}\033[0m")
                                    sys.stderr.write(line)
                                else:
                                    sys.stdout.write(line)
                except Exception as e:
                    print(f"Error reading output: {e}")
            
            with open(log_filepath, "w", encoding=utf8) as log_file:
                log_file.write(gap_line("Command Start")+endl*2)
                log_file.flush()
                
                result = subprocess.Popen(
                    cmd,
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    encoding=utf8,
                    bufsize=1,  # 行缓冲
                    universal_newlines=True
                )
                
                stdout_thread = threading.Thread(target=read_output, 
                                    args=(result.stdout, log_file, False))
                stderr_thread = threading.Thread(target=read_output, 
                                    args=(result.stderr, log_file, True))
                
                stdout_thread.start()
                stderr_thread.start()
                result.wait()
                stdout_thread.join()
                stderr_thread.join()
                
                log_file.write(endl+gap_line("Command End")+endl)
                log_file.write(f"\nReturn code: {result.returncode}\n")
        else:
            result = subprocess.run(
                cmd,
                env=env,
                check=False,
                # text=True,
                # encoding=utf8
            )

        # 检查返回码
        if result.returncode != 0:
            print(f"\n> Fail! return code: {result.returncode}")
            return False
        else:
            print(f'\n> Done!')
            return True


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
    max_samples:int = None
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
    preprocessing_batch_size:int = None
    streaming:bool = None
    max_steps:int = None
    buffer_size:int = None
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

    def start(self, cuda_visible, llama_factory_dir, log_to_file=True):
        llama_factory_dir = path(llama_factory_dir)
        assert llama_factory_dir.exists()
        os.chdir(llama_factory_dir)
        
        output_dir = path(self.output_dir); make_path(output_dir); print(output_dir)
        yaml_file = output_dir / f'sftlora.yaml'
        auto_dump(self.dict, yaml_file); print(f'> yaml file: {yaml_file}')
        log_filepath = output_dir / '_run.log' if log_to_file else None
        # cmd = ["llamafactory-cli", "train", str(yaml_file)]
        cmd = f'CUDA_VISIBLE_DEVICES={cuda_visible} llamafactory-cli train {str(yaml_file)} &> {str(log_filepath)}'
        # cmd = f'CUDA_VISIBLE_DEVICES={cuda_visible} llamafactory-cli train {str(yaml_file)} 2>&1 | tee {str(log_filepath)}'
        try:
            print('>', cmd)
            # exit()
            os.system(cmd)
            # self.run_cmd(
            #     cmd=cmd, 
            #     log_filepath=log_filepath, 
            #     env_dict={"CUDA_VISIBLE_DEVICES": cuda_visible},
            # )
        finally:
            make_path(file_path=path(self.output_dir)/'_end')


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

    def start(self, log_to_file=True):
        output_dir = path(self.export_dir); make_path(output_dir)
        time_str = Datetime_().format_str('%Y-%m-%d_%H-%M-%S')
        yaml_file = output_dir / f'mergelora_{time_str}.yaml'
        auto_dump(self.dict, yaml_file); print(f'> yaml file: {yaml_file}')
        log_filepath = output_dir / '_run.log' if log_to_file else None
        # cmd = ["llamafactory-cli", "export", str(yaml_file)]
        cmd = f'llamafactory-cli export {str(yaml_file)} &> {str(log_filepath)}'
        try:
            print('>', cmd)
            os.system(cmd)
            # self.run_cmd(
            #     cmd=cmd, log_filepath=log_filepath,
            # )
        finally:
            make_path(file_path=path(self.output_dir)/'_end')

