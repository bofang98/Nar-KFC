
from huggingface_hub import snapshot_download
from ..smp import *
from .video_base import VideoBaseDataset
from .utils import build_judge, DEBUG_MESSAGE

from scipy.spatial.distance import cdist
from scipy.linalg import eigh
import torch
import random
import json
from torch import linalg as LA
from dppy.finite_dpps import FiniteDPP
from pprint import pprint

FAIL_MSG = 'Failed to obtain answer via API.'



def unwrap_hf_pkl(pth, suffix='.mp4'):
    base_dir = os.path.join(pth, 'video_pkl/')
    target_dir = os.path.join(pth, 'video/')
    pickle_files = [os.path.join(base_dir, file) for file in os.listdir(base_dir)]
    pickle_files.sort()

    if not os.path.exists(target_dir):
        os.makedirs(target_dir, exist_ok=True)
        for pickle_file in pickle_files:
            with open(pickle_file, 'rb') as file:
                video_data = pickle.load(file)
            # For each video file in the pickle file, write its contents to a new mp4 file
            for video_name, video_content in video_data.items():
                output_path = os.path.join(target_dir, f'{video_name}{suffix}')
                with open(output_path, 'wb') as output_file:
                    output_file.write(video_content)
        print('The video file has been restored and stored from the pickle file.')
    else:
        print('The video file already exists.')


def load_json(fn):
    with open(fn, 'r') as f:
        data = json.load(f)
    return data



import numpy as np
def process_captions(captions, index_start, index_end, sample_num):
    caption_num = index_end - index_start - 1
    if caption_num < sample_num:    # cat all captions
        cat_caption = ' '.join(cap for cap in captions[index_start + 1: index_end])
    else:
        sample_indices = np.linspace(index_start + 1, index_end, sample_num).astype(int)
        cat_caption = ' '.join(captions[i] for i in sample_indices)
    
    # cat_caption = ' '.join(cap for cap in captions[index_start + 1: index_end])     # use all captions regardless of number

    # summarize captions
    # sum_prompt = ("You are given some language descriptions of several consecutive video clips. "
    #             "The descriptions are sequential and non-overlapping. "
    #             "Here are the descriptions: \n"
    #             f"{cat_caption}\n"
    #             "Please give me a no more than 50 words summary. "
    #             "When doing summarization, remember to keep any details that could be helpful for answering this question: \n"
    #             f"{question}")

    # messages = [{"role": "user", "content": sum_prompt},]
    # outputs = sum_model(messages, max_new_tokens=256,)
    # sum_text = outputs[0]["generated_text"][-1]['content']

    return cat_caption


def insert_cap_into_frames(message, frames, aux_data, fps_1_index):
    interval = fps_1_index[-1] - fps_1_index[0] - 1     # lasting time between
    max_num_cap = 30 * 7
    captions = aux_data['frame_caption']    # list 

    if interval <= max_num_cap:     # select all captions
        sel_captions = {i: captions[i] for i in range(fps_1_index[0]+1, fps_1_index[-1])}
    else:           # uniform select captions
        sample_indices = np.linspace(fps_1_index[0] + 1, fps_1_index[-1], max_num_cap).astype(int)
        sel_captions = {i: captions[i] for i in sample_indices}
    
    new_message = [{'type': 'text', 'value': ''}]
    new_message.append(dict(type='image', value=frames[0]))     # first frame

    for idx in range(1, len(fps_1_index)):
        start, end = fps_1_index[idx-1], fps_1_index[idx]
        middle_text = ' '.join(v for k, v in sel_captions.items() if start < k < end )
        if middle_text != '': 
            new_message.append(dict(type='text', value=middle_text))        # insert captions into frames
        new_message.append(dict(type='image', value=frames[idx]))
    return new_message


def ablate_cap_into_frames(message, frames, aux_data, fps_1_index, rank=0):
    interval = fps_1_index[-1] - fps_1_index[0] - 1     # lasting time between
    max_num_cap = 30 * 7
    captions = aux_data['frame_caption']    # list 

    if interval <= max_num_cap:     # select all captions
        sel_captions = {i: captions[i] for i in range(fps_1_index[0]+1, fps_1_index[-1])}
    else:           # uniform select captions
        sample_indices = np.linspace(fps_1_index[0] + 1, fps_1_index[-1], max_num_cap).astype(int)
        sel_captions = {i: captions[i] for i in sample_indices}
    
    message = []
    message.append(dict(type='image', value=frames[0]))     # first frame
    for idx in range(1, len(fps_1_index)):
        start, end = fps_1_index[idx-1], fps_1_index[idx]
        middle_text = ' '.join(v for k, v in sel_captions.items() if start < k < end )
        message.append(dict(type='text', value=middle_text))        # insert captions into frames
        message.append(dict(type='image', value=frames[idx]))


    # append narratives before and after the first and last keyframe
    # max_num_cap = 30 * 7
    # captions = aux_data['frame_caption']    # list
    # interval = len(captions)        # video length

    # if interval <= max_num_cap:     # select all captions
    #     sel_captions = {i: captions[i] for i in range(interval-1)}
    # else:           # uniform select captions
    #     sample_indices = np.linspace(0, interval-1, max_num_cap).astype(int)
    #     sel_captions = {i: captions[i] for i in sample_indices}

    # message = []
    # if fps_1_index[0]>0:    # beginning captions
    #     begin_cap = ' '.join(v for k, v in sel_captions.items() if 0 <= k < fps_1_index[0]) 
    #     message.append(dict(type='text', value=begin_cap))  

    # message.append(dict(type='image', value=frames[0]))     # first frame
    # for idx in range(1, len(fps_1_index)):
    #     start, end = fps_1_index[idx-1], fps_1_index[idx]
    #     middle_text = ' '.join(v for k, v in sel_captions.items() if start < k < end )
    #     message.append(dict(type='text', value=middle_text))        # insert captions into frames
    #     message.append(dict(type='image', value=frames[idx]))
    
    # if fps_1_index[-1]<interval-1:    # ending captions
    #     end_cap = ' '.join(v for k, v in sel_captions.items() if fps_1_index[-1] < k <= interval-1) 
    #     message.append(dict(type='text', value=end_cap))  

    return message



def top_k_sim_frames(vid, num_frames, query, judge_model, rank):
    processor, model = judge_model
    video_info = {
        'fps': vid.get_avg_fps(),
        'n_frames': len(vid),
    }
    fps_1_indices = np.arange(0, video_info['n_frames'], int(round(video_info['fps'])))
    video = [vid[i].asnumpy() for i in fps_1_indices[0:-1]]    
    video = [Image.fromarray(arr) for arr in video]

    sim_logits = {}
    for (index, frame) in zip(fps_1_indices[0:-1], video):
        inputs = processor(
            text=[query],
            images=[frame],
            return_tensors="pt",
            padding=True
        )
        for k, v in inputs.items(): inputs[k] = v.to("cuda:"+str(rank))
        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image.squeeze()     # 
        sim_logits[index] = logits_per_image.cpu()
    sorted_indices = sorted(sim_logits.items(), key=lambda x: x[1], reverse=True)
    return sorted([item[0] for item in sorted_indices[:num_frames]])


def dpp_sample(vid_clip_feat, query_clip_feat, k, w1=1.0, w2=0.0):
    n_frames = vid_clip_feat.shape[0]
    time_indices = torch.arange(n_frames, device=vid_clip_feat.device)

    vid_clip_feat = vid_clip_feat / torch.norm(vid_clip_feat, dim=1, keepdim=True)      # (n c)
    query_clip_feat = query_clip_feat / torch.norm(query_clip_feat, dim=1, keepdim=True)    # (1 c)
    frame_query = torch.matmul(vid_clip_feat, query_clip_feat.T)        # (n 1)     
    frame_frame = 1 - torch.matmul(vid_clip_feat, vid_clip_feat.T)      # (n n)
    frame_frame = frame_frame.clamp(0, 1) 

    time_diff = torch.abs(time_indices[:, None] - time_indices[None, :])
    temporal_diversity = (time_diff.float()**2 / n_frames**2)

    cross_modal_sim = torch.outer(frame_query.squeeze(-1), frame_query.squeeze(-1))
    content_diversity = (1 - frame_frame)
    L = w1 * cross_modal_sim * content_diversity + w2 * temporal_diversity      # kernel matrix
    L = (L + L.T) / 2       # ensure symmetric

    eigvals, eigvecs = LA.eigh(L)
    rank = torch.sum(eigvals > 1e-6).item()     # 秩
    eigvals = torch.clamp(eigvals, min=1e-6)  # 避免负特征值
    L = eigvecs @ torch.diag(eigvals) @ eigvecs.T

    # 添加对角线正则化项
    epsilon = 1e-3
    L += epsilon * torch.eye(L.shape[0], device=L.device)
    L = L.cpu().numpy()

    np.random.seed(42)
    dpp = FiniteDPP('likelihood', **{'L': L})
    k = min(k, rank)
    dpp.sample_exact_k_dpp(size=k)
    fps_1_indices = sorted(dpp.list_of_samples[-1])       # fps 1 indices
    return fps_1_indices





class VideoMME(VideoBaseDataset):

    MD5 = '85bdd91f9b29a99354c23b97ab7c113c'
    SYS = ''

    FRAMES_TMPL_NOSUB = """
These are the frames of a video. \
Select the best answer to the following multiple-choice question based on the video. \
Respond with only the letter (A, B, C, or D) of the correct option.
# """

#     FRAMES_TMPL_NOSUB = """
# These are the frames of a video and possible descriptions among those frames. \
# Select the best answer to the following multiple-choice question based on the video. \
# Respond with only the letter (A, B, C, or D) of the correct option.
# """

    FRAMES_TMPL_SUB = """
These are the frames of a video. \
This video's subtitles are listed below:
{}
Select the best answer to the following multiple-choice question based on the video. \
Respond with only the letter (A, B, C, or D) of the correct option.
"""

    TYPE = 'Video-MCQ'

    def __init__(self, dataset='Video-MME', use_subtitle=False):
        super().__init__(dataset=dataset)
        self.use_subtitle = use_subtitle
        self.dataset_name = dataset


    @classmethod
    def supported_datasets(cls):
        return ['Video-MME']

    def prepare_dataset(self, dataset_name='Video-MME', repo_id='lmms-lab/Video-MME'):

        def check_integrity(pth):
            data_file = osp.join(pth, f'{dataset_name}.tsv')

            if not os.path.exists(data_file):
                return False

            if md5(data_file) != self.MD5:
                return False
            data = load(data_file)
            for video_pth in data['video_path']:
                if not osp.exists(osp.join(pth, video_pth)):
                    return False
            return True

        cache_path = get_cache_path(repo_id)

        if cache_path is None:
            cache_path = os.environ.get('VIDEOMME_CACHE')
            
        if cache_path is not None and check_integrity(cache_path):
            dataset_path = cache_path
        else:

            def unzip_hf_zip(pth):
                import zipfile
                base_dir = pth
                target_dir = os.path.join(pth, 'video/')
                zip_files = [
                    os.path.join(base_dir, file) for file in os.listdir(base_dir)
                    if file.endswith('.zip') and file.startswith('video')
                ]
                zip_files.sort()

                if not os.path.exists(target_dir):
                    os.makedirs(target_dir, exist_ok=True)
                    for zip_file in zip_files:
                        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                            for member in zip_ref.namelist():
                                # Check if the member is a file (not a directory)
                                if not member.endswith('/'):
                                    # Extract the file to the specified directory
                                    source = zip_ref.open(member)
                                    target = open(os.path.join(target_dir, os.path.basename(member)), 'wb')
                                    with source, target:
                                        target.write(source.read())
                    print('The video file has been restored and stored from the zip file.')
                else:
                    print('The video file already exists.')

                subtitle_zip_file = os.path.join(base_dir, 'subtitle.zip')
                subtitle_target_dir = os.path.join(base_dir, 'subtitle')

                if not os.path.exists(subtitle_target_dir):
                    os.makedirs(subtitle_target_dir, exist_ok=True)
                    with zipfile.ZipFile(subtitle_zip_file, 'r') as zip_ref:
                        for member in zip_ref.namelist():
                            # Check if the member is a file (not a directory)
                            if not member.endswith('/'):
                                # Extract the file to the specified directory
                                source = zip_ref.open(member)
                                target = open(os.path.join(subtitle_target_dir, os.path.basename(member)), 'wb')
                                with source, target:
                                    target.write(source.read())
                    print('The subtitle file has been restored and stored from the zip file.')
                else:
                    print('The subtitle file already exists.')

            def generate_tsv(pth):

                data_file = osp.join(pth, f'{dataset_name}.tsv')
                if os.path.exists(data_file) and md5(data_file) == self.MD5:
                    return

                data_file = pd.read_parquet(os.path.join(pth, 'videomme/test-00000-of-00001.parquet'))
                data_file = data_file.assign(index=range(len(data_file)))
                data_file['video'] = data_file['videoID']
                data_file['video_path'] = data_file['videoID'].apply(lambda x: f'./video/{x}.mp4')
                data_file['subtitle_path'] = data_file['videoID'].apply(lambda x: f'./subtitle/{x}.srt')
                data_file['candidates'] = data_file['options'].apply(lambda x: x.tolist())

                data_file = data_file[['index', 'video', 'video_path', 'duration', 'domain', 'candidates',
                                       'sub_category', 'task_type', 'subtitle_path', 'question', 'answer']]

                data_file.to_csv(osp.join(pth, f'{dataset_name}.tsv'), sep='\t', index=False)

            if modelscope_flag_set():
                from modelscope import dataset_snapshot_download
                dataset_path = dataset_snapshot_download(dataset_id=repo_id)
            else:
                dataset_path = snapshot_download(repo_id=repo_id, repo_type='dataset')
            unzip_hf_zip(dataset_path)
            generate_tsv(dataset_path)

        data_file = osp.join(dataset_path, f'{dataset_name}.tsv')
        return dict(data_file=data_file, root=dataset_path)


    def save_video_frames(self, line, num_frames=8, fps=-1, video_llm=False, rank=0):
        video, query = line['video'], line['question']
        
        # read frames
        vid_path = osp.join(self.data_root, 'video', video + '.mp4')
        vid = decord.VideoReader(vid_path)
        video_info = {
            'fps': vid.get_avg_fps(),
            'n_frames': len(vid),
        }
        if num_frames > 0 and fps < 0:  
            step_size = len(vid) / (num_frames + 1)
            indices = [int(i * step_size) for i in range(1, num_frames + 1)]
            frame_paths = self.frame_paths(video, len(indices))
        elif fps > 0:
            # not constrained by num_frames, get frames by fps
            total_duration = video_info['n_frames'] / video_info['fps']
            required_frames = int(total_duration * fps)
            step_size = video_info['fps'] / fps
            indices = [int(i * step_size) for i in range(required_frames)]
            frame_paths = self.frame_paths_fps(video, len(indices), fps)

        
        flag = np.all([osp.exists(p) for p in frame_paths])

        if not flag:
            images = [vid[i].asnumpy() for i in indices]
            images = [Image.fromarray(arr) for arr in images]
            for im, pth in zip(images, frame_paths):
                if not osp.exists(pth):
                    im.save(pth)

        fps_1_frames_index = [int(i / video_info['fps']) for i in indices]
        return frame_paths, indices, video_info, fps_1_frames_index

    def graph_video_frames(self, frames, query, line, num_frames=8, fps=-1, video_llm=False, rank=0):
        video = line['video']
        frame_num = frames.shape[0]     # numbers of frames in fps=1

        vid_path = osp.join(self.data_root, 'video', video + '.mp4')
        vid = decord.VideoReader(vid_path)
        video_info = {
            'fps': vid.get_avg_fps(),
            'n_frames': len(vid),
        }

        if frame_num <= num_frames:     # less than num_frame seconds
            # do uniform sampling
            step_size = len(vid) / (num_frames + 1)
            indices = [int(i * step_size) for i in range(1, num_frames + 1)]
            frame_paths = self.frame_paths(video, len(indices))
        else:
            frames = frames / torch.norm(frames, dim=1, keepdim=True)      # (n c)
            query = query / torch.norm(query, dim=1, keepdim=True)    # (1 c)

            # frame-query
            frame_q = torch.matmul(frames, query.T)        # (n 1) 
            frame_query = frame_q.expand(frame_num, frame_num)      # (n n)
            
            # frame-frame
            gamma = 1.0
            frame_frame = torch.exp(-gamma * torch.matmul(frames, frames.T))        # (n n)
            
            # sum_score
            frame_score = frame_query + frame_frame        # (n n)
            frame_score = frame_score.triu(diagonal=1)      # (n n)
            frame_score.diagonal().copy_(frame_q.squeeze())     # (n n) for initialization

            torch.manual_seed(42)
            print(f"rank: {rank}\t fps1_frame_number: {frame_num}")

            shrink_K = 128 if line['duration'] == 'long' else frame_num // 4
            # shrink_K = 512 if frames.shape[0] > 64 else frame_num // 4
            fps1_indices, Score_svd = incremental_select_nodes(frame_score, num_frames, shrink_K, rank=rank)       # greey search 
            # fps1_indices = ILP_select_nodes(frame_score, num_frames, shrink_K, rank=rank)     # ILP search

            print(f"rank: {rank}\t sampled indices: {fps1_indices}")
            fps1_indices = local_search_indices(fps1_indices, Score_svd, search_radius=1, rank=rank)             # local search around frames (unsorted)
            unsorted_fps1_indices = fps1_indices

            assert len(fps1_indices) == num_frames
            fps1_indices = sorted(fps1_indices)
            # fps1_indices = [i+1 for i in fps1_indices]
            indices = [int(i * video_info['fps']) for i in fps1_indices]    # []
            frame_paths = self.frame_paths(video, len(indices))
            print(f'rank: {rank}\t sampled indices: {fps1_indices}')
        
        flag = np.all([osp.exists(p) for p in frame_paths])

        if not flag:
            images = [vid[i].asnumpy() for i in indices]
            images = [Image.fromarray(arr) for arr in images]
            for im, pth in zip(images, frame_paths):
                if not osp.exists(pth):
                    im.save(pth)

        fps_1_frames_index = [int(i / video_info['fps']) for i in indices]
        return frame_paths, indices, video_info, fps_1_frames_index



    def top_k_video_frames(self, frames, query, line, num_frames=8, fps=-1, video_llm=False, rank=0):
        video = line['video']
        frame_num = frames.shape[0]     # numbers of frames in fps=1

        vid_path = osp.join(self.data_root, 'video', video + '.mp4')
        vid = decord.VideoReader(vid_path)
        video_info = {
            'fps': vid.get_avg_fps(),
            'n_frames': len(vid),
        }

        if frame_num <= num_frames:     # less than num_frame seconds
            # do uniform sampling
            step_size = len(vid) / (num_frames + 1)
            indices = [int(i * step_size) for i in range(1, num_frames + 1)]
            frame_paths = self.frame_paths(video, len(indices))
        else:
            frames = frames / torch.norm(frames, dim=1, keepdim=True)      # (n c)
            query = query / torch.norm(query, dim=1, keepdim=True)    # (1 c)

            # frame-query
            frame_q = torch.matmul(frames, query.T).squeeze()        # (n 1) 
            unsort_fps1_indices = torch.topk(frame_q, num_frames).indices.tolist()     # []
            fps1_indices = sorted(unsort_fps1_indices)

            indices = [int(i * video_info['fps']) for i in fps1_indices]    # []
            frame_paths = self.frame_paths(video, len(indices))
            print(f'rank: {rank}\t sampled indices: {fps1_indices}')
        
        flag = np.all([osp.exists(p) for p in frame_paths])

        if not flag:
            images = [vid[i].asnumpy() for i in indices]
            images = [Image.fromarray(arr) for arr in images]
            for im, pth in zip(images, frame_paths):
                if not osp.exists(pth):
                    im.save(pth)

        fps_1_frames_index = [int(i / video_info['fps']) for i in indices]
        return frame_paths, indices, video_info, fps_1_frames_index, unsort_fps1_indices
    

    # def AKS_topk_frames(self, frames, query, line, num_frames=8, fps=-1, video_llm=False, rank=0):
    #     video = line['video']
    #     frame_num = frames.shape[0]     # numbers of frames in fps=1

    #     vid_path = osp.join(self.data_root, 'video', video + '.mp4')
    #     vid = decord.VideoReader(vid_path)
    #     video_info = {
    #         'fps': vid.get_avg_fps(),
    #         'n_frames': len(vid),
    #     }

    #     score = self.aks_frame_score[line['index']]
    #     # difference
    #     if len(score) > frame_num:
    #         score = score[:frame_num]
    #     else:
    #         score += [0]*(frame_num - len(score))
        
    #     print(f'difference between score and frame_num is {len(score)-frame_num}')

    #     _, fps1_indices = torch.topk(torch.tensor(score), num_frames)
    #     fps1_indices = sorted(fps1_indices)

    #     indices = [int(i * video_info['fps']) for i in fps1_indices]
    #     frame_paths = self.frame_paths(video, len(indices))
    #     print(f'rank: {rank}\t sampled indices: {fps1_indices}')

    #     flag = np.all([osp.exists(p) for p in frame_paths])

    #     if not flag:
    #         images = [vid[i].asnumpy() for i in indices]
    #         images = [Image.fromarray(arr) for arr in images]
    #         for im, pth in zip(images, frame_paths):
    #             if not osp.exists(pth):
    #                 im.save(pth)

    #     fps_1_frames_index = [int(i / video_info['fps']) for i in indices]
    #     return frame_paths, indices, video_info, fps_1_frames_index



    def save_video_into_images(self, line, num_frames=8):
        frame_paths, indices, video_info = self.save_video_frames(line['video'], num_frames)
        return frame_paths


    def build_prompt(self, line, num_frames, video_llm, fps, rank):
        if isinstance(line, int):
            assert line < len(self)
            line = self.data.iloc[line]
        
        vid_clip_feat = torch.from_numpy(
            np.load(osp.join(self.data_root, 'CLIP_video_emb', line['video']+'.npz'))['embeddings']).to("cuda:"+str(rank))     # (n 768)
        query_clip_feat = torch.from_numpy(
            np.load(osp.join(self.data_root, 'CLIP_QA_emb', str(line['index'])+'.npz'))['question_features']).to("cuda:"+str(rank))     # (1 768)

        import time

        # frames, indices, video_info, fps_1_frames_index = self.save_video_frames(line, num_frames, fps, video_llm, rank)  # uniform frame sampling
        frames, indices, video_info, fps_1_frames_index = self.graph_video_frames(vid_clip_feat, query_clip_feat, line, num_frames, fps, video_llm, rank)
        # frames, indices, video_info, fps_1_frames_index = self.top_k_video_frames(vid_clip_feat, query_clip_feat, line, num_frames, fps, video_llm, rank)
        
        if self.use_subtitle and os.path.exists(osp.join(self.data_root, line['subtitle_path'])):
            import pysubs2
            subs = pysubs2.load(osp.join(self.data_root, line['subtitle_path']), encoding='utf-8')
            subtitles = []

            for seleced_frame_id in indices:
                sub_text = ''
                cur_time = pysubs2.make_time(fps=video_info['fps'], frames=seleced_frame_id)
                for sub in subs:
                    if sub.start < cur_time and sub.end > cur_time:
                        sub_text = sub.text.replace('\\N', ' ')
                        break
                if sub_text.strip():
                    subtitles.append(sub_text)
            subtitles = '\n'.join(subtitles)
        else:
            subtitles = ''
        
        caption_file = osp.join(self.data_root, 'caption', line['video'] + '.json')
        # caption_embed = torch.from_numpy(
        #     np.load(osp.join(self.data_root, 'CLIP_caption_emb', line['video']+'.npz'))['caption_features']).to("cuda:"+str(rank))
        # caption_embed = caption_embed.squeeze()     # (n c)
        if os.path.exists(caption_file):     # TODO: add caption
            for k, v in load_json(caption_file).items(): 
                frame_cap = v        # [c1, c2, ...]
        

        ###### data collection
        aux_data = {
            'query_clip_feature': query_clip_feat,
            'video_clip_feature': vid_clip_feat,
            # 'caption_embedding': caption_embed,
            'frame_caption': frame_cap,
        }

        message = [dict(type='text', value=self.SYS)]

        # if video_llm:
        #     message.append(dict(type='video', value=osp.join(self.data_root, 'video', line['video'] + '.mp4')))
        # else:
        for im in frames:
            message.append(dict(type='image', value=im))
        # message = insert_cap_into_frames(message, frames, aux_data, fps_1_frames_index)
        message = ablate_cap_into_frames(message, frames, aux_data, fps_1_frames_index, rank)

        text_prompt = self.FRAMES_TMPL_NOSUB if not self.use_subtitle else self.FRAMES_TMPL_SUB.format(subtitles)
        message.append(dict(type='text', value=text_prompt))
        line['question'] += '\n' + '\n'.join(eval(line['candidates']))
        prompt = 'Question: {}\nAnswer: '.format(line['question'])
        message.append(dict(type='text', value=prompt))

        return message


    # It returns a dictionary
    @classmethod
    def evaluate(self, eval_file, **judge_kwargs):
        from .utils.videomme import get_dimension_rating, extract_characters_regex, extract_option

        assert eval_file.endswith('.xlsx'), 'data file should be an xlsx file'

        tmp_file = eval_file.replace('.xlsx', '_tmp.pkl')
        tgt_file = eval_file.replace('.xlsx', '_rating.json')
        score_file = eval_file.replace('.xlsx', '_score.xlsx')

        if not osp.exists(score_file):
            model = judge_kwargs.get('model', 'exact_matching')
            assert model in ['chatgpt-0125', 'exact_matching', 'gpt-4-0125']

            if model == 'exact_matching':
                model = None
            elif gpt_key_set():
                model = build_judge(**judge_kwargs)
                if not model.working():
                    warnings.warn('OPENAI API is not working properly, will use exact matching for evaluation')
                    warnings.warn(DEBUG_MESSAGE)
                    model = None
            else:
                warnings.warn('OPENAI_API_KEY is not set properly, will use exact matching for evaluation')
                model = None
            res = {} if not osp.exists(tmp_file) else load(tmp_file)
            res = {k: v for k, v in res.items() if FAIL_MSG not in v}

            data = load(eval_file)
            data_un = data[~pd.isna(data['prediction'])]

            for idx in data['index']:
                ans = data.loc[data['index'] == idx, 'answer'].values[0]
                pred = str(data.loc[data['index'] == idx, 'prediction'].values[0])

                if extract_characters_regex(pred) == '':
                    extract_pred = extract_option(
                        model,
                        data.loc[data['index'] == idx].to_dict(orient='records')[0],
                        'Video-MME'
                    )
                    data.loc[idx, 'score'] = int(extract_pred == ans)
                else:
                    data.loc[idx, 'score'] = int(extract_characters_regex(pred) == ans)

            rejected = [x for x in data['score'] if x == -1]

            print(
                f'Among {len(data)} questions, failed to obtain prediction for {len(data) - len(data_un)} questions, '
                f'failed to obtain the score for another {len(rejected)} questions. '
                f'Those questions will be counted as -1 score in ALL rating, and will not be counted in VALID rating.'
            )

            dump(data, score_file)

        rating = get_dimension_rating(score_file)
        dump(rating, tgt_file)
        return rating
