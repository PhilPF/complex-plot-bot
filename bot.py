#IMPORTS

import tweepy as tp
import os
import time

import wget

import numpy as np
import pylab as plt
import sympy as sp
from colorsys import hls_to_rgb
import random 
from PIL import Image
import PIL.ImageOps 



#FUNCTIONS

# Operations


def suma(a,b,ta,tb,la,lb):
    return(a+b, '('+ta+')+('+tb+')', la+'+'+lb)
    
def resta(a,b,ta,tb,la,lb):
    return(a-b, '('+ta+')-('+tb+')', la+'-'+lb)

def prod(a,b,ta,tb,la,lb):
    return(a*b, '('+ta+')*('+tb+')', la+r'\cdot'+lb)
    
def div(a,b,ta,tb,la,lb):
    return(a/b, '('+ta+')/('+tb+')',r'\frac{'+la+'}{'+lb+'}')
    
def pot(a,b,ta,tb,la,lb):
    return(a**b, '('+ta+')^('+tb+')', '{'+la+'}^{'+lb+'}')
    
def potinv(a,b,ta,tb,la,lb):
    return(a**(1/b), '('+ta+')^(1/('+tb+'))', la+r'^\frac{1}{'+lb+'}')
    

# Fundamental

def exp(a,t,l):
    return (np.exp(a),'exp('+t+')', 'e^'+l)

def log(a,t,l):
    return (np.log(a),'log('+t+')', r'\log('+l+')')


#def absol(a,t,l):
#    return (np.abs(a), '\|'+t+'\|', '|'+l+'|')

def inv(a,t,l):
    return (1/a, '(1/('+t+'))', r'\frac{1}{'+l+'}')

def power(a,t,l):
    p = random.randrange(2,5)
    return (np.power(a,p), '('+t+')^'+str(p), '{'+l+'}^{'+str(p)+'}')


# Trigonometric

def sin(a,t,l):
    return (np.sin(a), 'sin('+t+')', r'\sin('+l+')')

def cos(a,t,l):
    return (np.cos(a), 'cos('+t+')', r'\cos('+l+')')

def tan(a,t,l):
    return (np.tan(a), 'tan('+t+')', r'\tan('+l+')')

def sinh(a,t,l):
    return (np.sinh(a), 'sinh('+t+')', r'\sinh('+l+')')

def cosh(a,t,l):
    return (np.cosh(a), 'cosh('+t+')', r'\cosh('+l+')')

def tanh(a,t,l):
    return (np.tanh(a), 'tanh('+t+')', r'\tanh('+l+')')

def arcsin(a,t,l):
    return (np.arcsin(a), 'arcsin('+t+')', r'\arcsin('+l+')')

def arccos(a,t,l):
    return (np.arccos(a), 'arccos('+t+')', r'\arccos('+l+')')

def arctan(a,t,l):
    return (np.arctan(a), 'arctan('+t+')', r'\arctan('+l+')')

def arcsinh(a,t,l):
    return (np.arcsinh(a), 'arcsinh('+t+')', r'\operatorname{arcsinh}('+l+')')

def arccosh(a,t,l):
    return (np.arccosh(a), 'arccosh('+t+')', r'\operatorname{arccosh}('+l+')')

def arctanh(a,t,l):
    return (np.arctanh(a), 'arctanh('+t+')', r'\operatorname{arctanh}('+l+')') 

#CREDENTIALS

API_KEY = os.environ.get("API_KEY")
API_KEY_SECRET = os.environ.get("API_KEY_SECRET")

ACCESS_TOKEN = os.environ.get("ACCESS_TOKEN")
ACCESS_TOKEN_SECRET = os.environ.get("ACCESS_TOKEN_SECRET")


#PLOT GENERATOR

def colorize(z):
    n,m = z.shape
    c = np.zeros((n,m,3))
    c[np.isinf(z)] = (1.0, 1.0, 1.0)
    c[np.isnan(z)] = (0.5, 0.5, 0.5)

    idx = ~(np.isinf(z) + np.isnan(z))
    A = (np.angle(z[idx]) + np.pi) / (2*np.pi)
    A = (A + 0.5) % 1.0
    B = 1.0 - 1.0/(1.0+abs(z[idx])**0.3)
    c[idx] = [hls_to_rgb(a, b, 0.8) for a,b in zip(A,B)]
    return c

def RandomFunction():
    opList1=[suma, resta]
    opList2=[prod, div, pot, potinv]
    opList=[opList1, opList2]
    
    fList1=[exp, log]
    fList2=[sin, arcsin, cos, arccos, tan, arctan, sinh, arcsinh, cosh, arccosh, tanh, arctanh]
    fList3=[inv, power]
    fList=[fList1, fList2, fList3]
    
    newfType=random.choice(fList)

    function=random.choice(newfType)(Z,'z','z')
    oldOpType=[]
    oldfType=newfType
    for i in range(2,random.randint(4,6)):
        while True:
            newOpType=random.choice(opList)
            if newOpType!=oldOpType: 
                oldOpType=newOpType
                break 
        while True:
            newfType=random.choice(fList)
            if newfType!=oldfType: 
                oldfType=newfType
                break 
        newf = random.choice(newfType)(Z,'z','z')
        function = random.choice(newOpType)(function[0],newf[0],function[1],newf[1],function[2],newf[2])
        
    return function

def GenerateLayout(im1, im2, resample=Image.BICUBIC, margin=200):
    scale_factor=im2.width/im2.height
    new_im2_height=int(im1.height/4)
    new_im2_width=int(new_im2_height*scale_factor)
    phantom_margin_h=im1.width-new_im2_width
    
    phantom_margin_v=0
    if new_im2_width > im1.width:
        phantom_margin_h=2*margin
        phantom_margin_v = new_im2_height
        new_im2_width=im1.width-2*margin
        new_im2_height=int(new_im2_width/scale_factor)
        phantom_margin_v -= new_im2_height
        
    else :
        phantom_margin_v = new_im2_height
        new_im2_height=int(0.7*new_im2_height)
        new_im2_width=int(new_im2_height*scale_factor)
        phantom_margin_v -= new_im2_height
        phantom_margin_h = im1.width-new_im2_width

    new_im2 = im2.resize((new_im2_width, new_im2_height), resample=resample)
    dst = Image.new('RGB', (im1.width, im1.height + new_im2_height + 2*margin + phantom_margin_v), color=(0,0,0))
    dst.paste(im1, (0, 0))
    dst.paste(new_im2, (int(phantom_margin_h/2), im1.height + margin + int(phantom_margin_v/2)))
    return dst

def plot(i):
    
    plot_name='plot_'+str(i)+'.png'
    
    random.seed(time.time())

    N = 1000
    A = np.zeros((N,N),dtype='complex')
    axis_x = np.linspace(-5,5,N)
    axis_y = np.linspace(-5,5,N)
    X,Y = np.meshgrid(axis_x,axis_y)
    global Z
    Z = X + Y*1j
    
    function = RandomFunction()

    A=function[0]

    # Plot the array "A" using colorize
    plt.imshow(colorize(A), interpolation='none',extent=(-5,5,-5,5))

    plt.axis('off')
    plt.subplots_adjust(0,0,1,1,0,0)
    plt.margins(0,0)
    plt.savefig('plots/image/'+plot_name, pad_inches = 0, dpi=1200, bbox_inches='tight')

    print(sp.simplify(function[1]))
    sp.preview(sp.simplify(function[1]), viewer='file', filename='plots/latex/'+plot_name, euler=False,  dvioptions=["-T", "tight", "-z", "0", "--truecolor", "-D 300"] )

    plot_graph=Image.open('plots/image/'+plot_name)
    plot_latex=Image.open('plots/latex/'+plot_name)
    
    plot_latex = PIL.ImageOps.invert(plot_latex)
    
    GenerateLayout(plot_graph, plot_latex).save('plots/layout/'+plot_name)


#PROFILE PICTURE UPDATER

def editProfPic():
    downloaded=Image.open('profilePic/profilePic.jpg')
    
    logo=Image.open('profilePic/logo.png')
    logo = logo.resize((downloaded.width, downloaded.width), resample=Image.BICUBIC)

    profPic = Image.new('RGB', (downloaded.width, downloaded.width), color=(0,0,0))
    profPic.paste(downloaded, (0, 0))
    profPic.paste(logo, (0,0), logo)
    profPic.save('profilePic/profilePic.jpg')

def editBanner():
    banner=Image.open('profilePic/banner_logo.png')
    
    downloaded=Image.open('profilePic/profilePic.jpg')
    downloaded = downloaded.resize((banner.width, int(banner.width*downloaded.height/downloaded.width)), resample=Image.BICUBIC)

    profPic = Image.new('RGB', (banner.width, banner.height), color=(0,0,0))
    profPic.paste(downloaded, (0, -int(downloaded.width/3)))
    profPic.paste(banner, (0,0), banner)
    profPic.save('profilePic/banner.jpg')

def updateProfPic():
    maxLikesFile = open('profilePic/maxLikes.txt', 'r')
    maxLikes = int(maxLikesFile.read())
    maxLikesFile.close()    
    
    # login to twitter account api
    auth = tp.OAuthHandler(API_KEY, API_KEY_SECRET)
    auth.set_access_token(ACCESS_TOKEN, ACCESS_TOKEN_SECRET)
    api = tp.API(auth)
    
    tweet = api.user_timeline(screen_name='complex_plot', count=20, include_rts=False, exclude_replies=True)
    
    media_files = list()
    for status in reversed(tweet):
        media = status.entities.get('media', [])
        #print(status.favorite_count)
        if(len(media) > 0 and status.favorite_count>=maxLikes):
            media_files.append(media[0]['media_url'])
            maxLikes=status.favorite_count
            newProfPicID=str(status.id)
            #print(maxLikes)
    
    if (len(media_files)>0):
        media_file=media_files.pop()
        
        maxLikesIDFile = open('profilePic/maxLikesID.txt', 'r')
        maxLikesID = str(maxLikesIDFile.read())
        maxLikesIDFile.close() 
        
        if(newProfPicID!=maxLikesID):
            print("New Profile Picture !!")
            
            with os.scandir('profilePic') as entries:
                for entry in entries:
                    if entry.is_file():
                        if(entry.name == "profilePic.jpg"):
                            os.remove(entry)
            wget.download(media_file, "profilePic/profilePic.jpg")
            
            maxLikesFile = open('profilePic/maxLikes.txt', 'w')
            maxLikesFile.write(str(maxLikes))
            maxLikesFile.close() 
            
            maxLikesIDFile = open('profilePic/maxLikesID.txt', 'w')
            maxLikesIDFile.write(newProfPicID)
            maxLikesIDFile.close()
            
            editBanner()
            editProfPic()
            
            api.update_profile_image('profilePic/profilePic.jpg')
            api.update_profile_banner('profilePic/banner.jpg')


#PLOT UPLOADER

j=1
index=0

def gen():
    
    ifile = open('plots/index.txt', 'r')
    
    i = int(ifile.read())
    
    global index
    index = i
    
    ifile.close()
    
    
    plot(i)
    
    
    ifile=open('plots/index.txt', 'w')
    
    ifile.write(str(i+1))
    
    ifile.close()



# login to twitter account api
auth = tp.OAuthHandler(API_KEY, API_KEY_SECRET)
auth.set_access_token(ACCESS_TOKEN, ACCESS_TOKEN_SECRET)
api = tp.API(auth)


for i in range(0,j):
    s_time = time.time()

    updateProfPic()

    gen()
    
    with os.scandir('plots/layout') as entries:
        for entry in entries:
            if entry.is_file():
                if(entry.name == 'plot_'+str(index)+'.png'):
                    media = api.media_upload(entry)
                    api.update_status(status='', media_ids=[media.media_id])
                    
                    print("--- %s seconds ---" % round(time.time() - s_time,2))        
        
