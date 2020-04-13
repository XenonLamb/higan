import os.path
import argparse
import numpy as np
from tqdm import tqdm
import torch
import _io
import streamlit as st
import altair as alt
import pandas as pd


from models.helper import build_generator
from utils.logger import setup_logger
from utils.editor import parse_boundary_list
from utils.editor import get_layerwise_manipulation_strength
from utils.editor import manipulate
from utils.visualizer import HtmlPageVisualizer
from utils.visualizer import fuse_images
from utils.visualizer import VideoWriter
from utils.visualizer import save_image
from models.model_settings import MODEL_POOL



_ATTRIBUTE_LIST_DESCRIPTION = '''
Attribute list desctipiton:

  Attribute list should be like:

    (age, z): $AGE_BOUNDARY_PATH
    (gender, w): $GENDER_BOUNDARY_PATH
    DISABLE(pose, wp): $POSE_BOUNDARY_PATH

  where the pose boundary from WP space will be ignored.
'''

## utility to handle latent code state
GAN_HASH_FUNCS = {
    _io.TextIOWrapper : id,
torch.nn.backends.thnn.THNNFunctionBackend:id,
torch.nn.parameter.Parameter:id,
torch.Tensor:id,
}

@st.cache(allow_output_mutation=True,suppress_st_warning=True,show_spinner=False)
def parse_args():
  """Parses arguments."""
  parser = argparse.ArgumentParser(
      description='Manipulate images from latent space of GAN.',
      epilog=_ATTRIBUTE_LIST_DESCRIPTION,
      formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument('model_name', type=str,default='stylegan_bedroom',
                      help='Name of the model used for synthesis.')
  parser.add_argument('boundary_name', type=str,
                      help='Name of the boundary to manipulate.')
  parser.add_argument('-c', '--latent_codes_path', type=str, default='',
                      help='If specified, will load latent codes from given '
                           'path instead of randomly sampling. (default: None)')
  parser.add_argument('--latent_space_type', type=str, default='w',
                      choices=['z', 'w', 'wp'],
                      help='Space type of the input latent codes. This field '
                           'will also be used for latent codes sampling if '
                           'needed. (default: `w`)')
  parser.add_argument('--manipulate_layers', type=str, default='6-11',
                      help='Indices of the layers to perform manipulation. '
                           'Active ONLY when `layerwise_manipulation` is set '
                           'as `True`. If not specified, all layers will be '
                           'manipulated. More than one layers should be '
                           'separated by `,`. (default: None)')
  return parser.parse_args(['boundary_name','indoor_lighting'])

class LatentState:
    pass

@st.cache(allow_output_mutation=True,hash_funcs=GAN_HASH_FUNCS,show_spinner=False)
def fetch_session(model,kwargs):
    session = LatentState()
    session.latent = init_latent(model,kwargs)
    return session

@st.cache(allow_output_mutation=True,suppress_st_warning=True,show_spinner=False)
def prepare_boudary(model_name, boundary_name, latent_space_type=None):
    boundary_load_state = st.text('Loading boundary...')
    basepath = './boundaries/'
    basepath= basepath+ model_name+'_'
    basepath = basepath + boundary_name + '_'
    if latent_space_type=='w':
        basepath+='w_'
    basepath+='boundary.npy'
    if not os.path.isfile(basepath):
        raise ValueError(f'Boundary `{basepath}` does not exist!')
    boundary = np.load(basepath)
    boundary_load_state.empty()
    return boundary


@st.cache(allow_output_mutation=True,hash_funcs=GAN_HASH_FUNCS,suppress_st_warning=True,show_spinner=False)
def load_model(model_name,logger=None):
    model_load_state = st.text('Loading GAN model...')
    model = build_generator(model_name, logger=logger)
    model_load_state.empty()
    return model


## randomly initialize latent code
def init_latent(model,latent_space_type):
    return model.easy_sample(num=1,
                                 latent_space_type=latent_space_type)


## update the latent code from uploaded file
@st.cache(allow_output_mutation=True,hash_funcs=GAN_HASH_FUNCS,suppress_st_warning=True,show_spinner=False)
def load_latent(latent_file, latent_state, model, latent_space_type):
    latent_codes = np.load(latent_file)
    latent_codes = model.preprocess(latent_codes=latent_codes,
                                    latent_space_type=latent_space_type)
    latent_state.latent = latent_codes
    return


@st.cache(allow_output_mutation=True,hash_funcs=GAN_HASH_FUNCS,suppress_st_warning=True,show_spinner=False)
def get_logger(logger_name):

    work_dir = f'simple_manipulation_result'
    if not os.path.exists(work_dir):
        os.mkdir(work_dir)
    logger = setup_logger(work_dir, '', logger_name)
    return logger

def model_handler(args):
    #model_name = st.text_input('GAN model name:', args.model_name)
    model_name = st.sidebar.selectbox(
        'Which GAN model?',
        ['stylegan_bedroom', 'stylegan_bridge','stylegan_churchoutdoor','stylegan_kitchen','stylegan_livingroom','stylegan_tower'])
    if model_name in MODEL_POOL:
        args.model_name = model_name

    gan_type = MODEL_POOL[args.model_name]['gan_type']

    if gan_type == 'stylegan':
        latent_space_type = st.sidebar.selectbox(
            'Which type of latent space?',
            [ 'w', 'wp', 'z'])
        args.latent_space_type = latent_space_type


def boundary_handler(args):
    boundary_name = st.sidebar.text_input('Boundary name:', 'indoor_lighting')
    #manipulate_layers_name = st.sidebar.text_input('Manipulating layers:', '6-11')
    #if manipulate_layers_name is not None:
    #    args.manipulate_layers = manipulate_layers_name
    boundary_path = f'boundaries/{args.model_name}/{boundary_name}_boundary.npy'
    if os.path.exists(boundary_path):
        boundary, manipulate_layers = load_boundary(boundary_path, args.manipulate_layers)
    else:
        tx = st.text('Boundary file does not exist!')
        return None, None

    return boundary, manipulate_layers


@st.cache(allow_output_mutation=True,hash_funcs=GAN_HASH_FUNCS,suppress_st_warning=True,show_spinner=False)
def load_boundary(boundary_path,manipulate_layers):
    try:
        print('loading boundary')
        boundary_file = np.load(boundary_path, allow_pickle=True).item()
        boundary = boundary_file['boundary']
        manipulate_layers = boundary_file['meta_data']['manipulate_layers']
    except ValueError:
        boundary = np.load(boundary_path)
        manipulate_layers = manipulate_layers
    return boundary, manipulate_layers

@st.cache(allow_output_mutation=True,hash_funcs=GAN_HASH_FUNCS,suppress_st_warning=True,show_spinner=False)
def get_full_latent(model,latents,latent_space_type):
    latent_load_state = st.text('Preparing latent codes...')
    latent_codes = model.easy_synthesize(latent_codes=latents,
                                         latent_space_type=latent_space_type,
                                         generate_style=False,
                                         generate_image=False)
    latent_load_state.empty()
    return latent_codes

## set page to wide mode
def _max_width_():
    max_width_str = f"max-width: 2000px;"
    st.markdown(
        f"""
    <style>
    .reportview-container .main .block-container{{
        {max_width_str}
    }}
    </style>
    """,
        unsafe_allow_html=True,
    )


_max_width_()

st.title('HiGAN Interactive Demo')

args = parse_args()

work_dir = f'simple_manipulation_result'
#logger_name = f'{args.model_name}_manipulation_logger'
logger = get_logger('logger')

model_handler(args)

model = load_model(args.model_name)

latent_codes_state = fetch_session(model,args.latent_space_type)

if st.sidebar.button('Get random latent code!'):
    latent_codes_state.latent = init_latent(model,args.latent_space_type)

uploaded_code = st.sidebar.file_uploader("Or upload a latent code .npz file", type="npy")
if uploaded_code is not None:
    load_latent(uploaded_code, latent_codes_state, model, args.latent_space_type)


total_num = latent_codes_state.latent.shape[0]

latent_codes = get_full_latent(model, latent_codes_state.latent,args.latent_space_type)

for key, val in latent_codes.items():
    np.save(os.path.join(work_dir, f'{key}.npy'), val)


boundary, manipulate_layers = boundary_handler(args)
if boundary is not None:
    #np.save(os.path.join(work_dir, f'{prefix}_boundary.npy'), boundary)
    stepsize = st.sidebar.slider('Manipulation step', -5.0, 5.0,0., 0.1)

    strength = get_layerwise_manipulation_strength(
        model.num_layers, model.truncation_psi, model.truncation_layers)

    codes = manipulate(latent_codes=latent_codes['wp'],
                         boundary=boundary,
                         start_distance=0.,
                         end_distance=0.+stepsize,
                         step=2,
                         layerwise_manipulation=True,
                         num_layers=model.num_layers,
                         manipulate_layers=manipulate_layers,
                         is_code_layerwise=True,
                         is_boundary_layerwise=False,
                         layerwise_manipulation_strength=strength)
    #np.save(os.path.join(work_dir, f'{prefix}_manipulated_wp.npy'), codes)
    out_images = []
    for s in range(2):
        images = model.easy_synthesize(codes[:, s], latent_space_type='wp')['image']
        for n, image in enumerate(images):
            out_images.append(image)

    st.image(out_images,width=512, caption=['Source face','Manipulated face'])
