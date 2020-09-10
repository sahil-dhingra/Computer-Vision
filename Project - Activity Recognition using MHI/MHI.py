import numpy as np
import cv2
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.externals.joblib import dump, load
import os

IMG_DIR = "input_images"
VID_DIR = "input_videos"
OUT_DIR = "output"

class Hu_moment:
   
    def __init__(self, image_t_1, theta, k_size, tao, mei = 0):
        self.theta = theta
        self.k_size = k_size
        self.tao = tao
        self.mei = mei
        
        h, w = image_t_1.shape
        self.m_t_1 = np.zeros((h, w))
        self.m_t   = np.zeros((h, w))
        
    def frame_difference(self, image_t, image_t_1):
        theta = self.theta
        k_size = self.k_size
        image_difference = image_t - image_t_1
        frame_difference = 1*(image_difference > theta)
        kernel = np.ones((k_size, k_size), np.uint8)
        morph_open = cv2.morphologyEx(1.*frame_difference, cv2.MORPH_OPEN, kernel)
        return morph_open
    
    def mhi(self, image_t, image_t_1):
        tao = self.tao
        mei = self.mei
        b_t = self.frame_difference(image_t, image_t_1)
        h, w = image_t.shape
        self.m_t_1 = self.m_t
        m_t = np.zeros((h,w))
        ind_1 = np.where(b_t==1)
        ind_0 = np.where(b_t==0)
        m_t[ind_1] = tao
        m_t[ind_0] = np.where(self.m_t_1[ind_0] - 1 > 0, self.m_t_1[ind_0] - 1, 0)
        if mei == 1:
            m_t = 1*(m_t>0)
        self.m_t = m_t
        return m_t
        
    def moments(self, image, i, j, x_avg = 0, y_avg = 0):
        h, w = image.shape
        x = np.arange(0, h, 1) - x_avg
        y = np.arange(0, w, 1) - y_avg
        moment = np.sum((image.T*(x**i)).T*(y**j))
        return moment
        
    def hu(self, image_t, image_t_1, p, q):
        mhi = 1.*self.mhi(image_t, image_t_1)
        if mhi.max()>0:
            mhi /= mhi.max() ## Normalize
        x_avg = self.moments(mhi, 1, 0)/self.moments(mhi, 0, 0)
        y_avg = self.moments(mhi, 0, 1)/self.moments(mhi, 0, 0)
        
        u_pq = self.moments(mhi, p, q, x_avg, y_avg)
        n_pq = u_pq/(self.moments(mhi, 0, 0, x_avg, y_avg))**(1 + 0.5*(p+q))
        
        return (u_pq, n_pq)
    
    
def get_moments(HU, image_t, image_t_1):
    u20, n20 = HU.hu(image_t, image_t_1, 2, 0)
    u02, n02 = HU.hu(image_t, image_t_1, 0, 2)
    u11, n11 = HU.hu(image_t, image_t_1, 1, 1)
    u30, n30 = HU.hu(image_t, image_t_1, 3, 0)
    u03, n03 = HU.hu(image_t, image_t_1, 0, 3)
    u12, n12 = HU.hu(image_t, image_t_1, 1, 2)
    u21, n21 = HU.hu(image_t, image_t_1, 2, 1)
    
    u1 = u20 + u02
    u2 = (u20 - u02)**2 + 4*(u11)**2
    u3 = (u30 - 3*u12)**2 + (3*u21 - u03)**2
    u4 = (u30 + u12)**2 + (u21 + u03)**2
    u5 = (u30 - 3*u12)*(u30 + u12)*((u30 + u12)**2 - 3*(u21 + u03)**2) + \
            (3*u21 - u03)*(u21 + u03)*(3*(u30 + u12)**2 - (u21 + u03)**2)
    u6 = (u20 - u02)*((u30 + u12)**2 - (u21 + u03)**2) + 4*u11*(u30 + u12)*(u21 + u03)
    u7 = (3*u21 - u03)*(u30 + u12)*((u30 + u12)**2 - 3*(u21 + u03)**2) + \
            (u30 - 3*u12)*(u21 + u03)*(3*(u30 + u12)**2 - (u21 + u03)**2)

    h1 = n20 + n02
    h2 = (n20 - n02)**2 + 4*(n11)**2
    h3 = (n30 - 3*n12)**2 + (3*n21 - n03)**2
    h4 = (n30 + n12)**2 + (n21 + n03)**2
    h5 = (n30 - 3*n12)*(n30 + n12)*((n30 + n12)**2 - 3*(n21 + n03)**2) + \
            (3*n21 - n03)*(n21 + n03)*(3*(n30 + n12)**2 - (n21 + n03)**2)
    h6 = (n20 - n02)*((n30 + n12)**2 - (n21 + n03)**2) + 4*n11*(n30 + n12)*(n21 + n03)
    h7 = (3*n21 - n03)*(n30 + n12)*((n30 + n12)**2 - 3*(n21 + n03)**2) + \
            (n30 - 3*n12)*(n21 + n03)*(3*(n30 + n12)**2 - (n21 + n03)**2)
            
    return (u1, u2, u3, u4, u5, u6, u7, h1, h2, h3, h4, h5, h6, h7)


def mp4_video_writer(filename, frame_size, fps=20):
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    return cv2.VideoWriter(filename, fourcc, fps, frame_size)


def save_image(filename, image):
    cv2.imwrite(os.path.join(OUT_DIR, filename), image)


def video_frame_generator(filename):
    """A generator function that returns a frame on each 'next()' call."""
    video =  cv2.VideoCapture(filename)

    while video.isOpened():
        ret, frame = video.read()

        if ret:
            yield frame
        else:
            break

    video.release()
    yield None


def get_data(folder, video_file, frames, theta, k_size, tao):
    data = []
    
    video = os.path.join(VID_DIR, folder, video_file)
    image_gen = video_frame_generator(video)
    image_t_1 = image_gen.__next__()
    i = 1
    image_t   = image_gen.__next__()
    i = 2
    
    image_t_1 = cv2.cvtColor(image_t_1, cv2.COLOR_BGR2GRAY)
    image_t   = cv2.cvtColor(image_t,   cv2.COLOR_BGR2GRAY)
    
    HU_0 = Hu_moment(image_t_1, theta, k_size, tao, mei = 0)
    HU_1 = Hu_moment(image_t_1, theta, k_size, tao, mei = 1)
    
    while image_t is not None:
        if i in frames:
            data.append(np.array([get_moments(HU_0, image_t, image_t_1), 
                                  get_moments(HU_1, image_t, image_t_1)]).flatten())
        image_t_1 = image_t.copy()
        image_t   = image_gen.__next__() 
        i = i + 1
        if image_t is not None:
            image_t   = cv2.cvtColor(image_t,   cv2.COLOR_BGR2GRAY)

    data = np.array(data)
    return data


def generate_data(sample):
    # Get frame list for each video
    frames = open("00sequences.txt")
    frame_list = frames.read()
    frames.close()
    frame_list = frame_list.split('\n')
    video_frames = {}
    for i in frame_list:
        if i!='':
            video = i.split('\t')
            frame_range = video[-1].split(',')
            n = len(frame_range)
            frame_cuts =  [frame_range[j].strip().split('-') for j in range(n)]
            frame_index = []
            for j in range(n):
                frame_index=frame_index + list(range(int(frame_cuts[j][0]), int(frame_cuts[j][1])+1))
            video_frames[video[0].strip()+'_uncomp.avi'] = frame_index

    activity = ['walking', 'jogging', 'running', 'boxing', 'handwaving', 'handclapping']
    params = {'walking': (5, 4, 30), 'jogging': (5, 5, 3), 'running': (5, 6, 1),\
              'boxing': (3, 6, 120), 'handwaving': (5, 5, 30), 'handclapping': (5, 5, 2)}

    Y = []
    X = np.zeros((1,28))
    for i in activity:
        path = os.path.join(VID_DIR, i)
        filenames = next(os.walk(path))[2]
        np.random.seed(21)
        theta, k_size, tao = params[i]
        filenames = [i for j in sample for i in filenames if j in i]
        for j in filenames:
            print(i, j)
            X_ = get_data(i, j, video_frames[j], theta, k_size, tao)
            X = np.vstack([X, X_])
            n = X_.shape[0]
            Y.append([i]*n)
    
    x = X[1:,:]
    x = np.array(x)
    X = x[~np.isnan(x).any(axis=1)]
    y = np.array([item for sublist in Y for item in sublist])
    Y = y[~np.isnan(x).any(axis=1)]
    
    Y = pd.factorize(Y)
    Y = Y[0]
    
    return (X,Y)


def model_samples():
    persons = ['person{}'.format(f"{i:02d}") for i in range(1,26)]
    train_persons = persons[10:18]
    val_persons = persons[18:25] + [persons[0], persons[3]]
    val_persons.remove('person22')
    train_val = train_persons + val_persons
    test_persons = [i for i in persons if i not in train_val]
    return (train_persons, val_persons, test_persons)


def train_model(train_persons, n_estimators=1200, min_samples_leaf = 10, subsample=0.7, max_depth=5):
    X, Y = generate_data(train_persons)

    gb = GradientBoostingClassifier(n_estimators = n_estimators, min_samples_leaf = min_samples_leaf, \
                                    subsample = subsample, max_depth = max_depth)
    print('Training model..')
    gb.fit(X, Y)
    pred_train = gb.predict(X)
    return X, Y, gb


def val_model(val_persons, model):
    X, Y = generate_data(val_persons)
    pred_val = gb.predict(X)
    return (X, Y)


def test_model(test_persons, model):
    X, Y = generate_data(test_persons)
    pred_test = gb.predict(X)
    print(confusion_matrix(Y, pred_test))
    return (X, Y)


def run_model_processes():
    train_persons, val_persons, test_persons = model_samples()
    
    X_train, y_train, gb = train_model(train_persons)
    X_val, y_val = val_model(val_persons, gb)
    X_test, y_test = test_model(test_persons, gb)
    
    X_train_val = np.vstack([X_train,X_val])
    y_train_val = np.append(y_train,y_val)
    gb2 = GradientBoostingClassifier(n_estimators=1200, min_samples_leaf = 10, subsample=0.7, max_depth=5)
    gb2.fit(X_train_val, y_train_val)
    pred_train_val = gb2.predict(X_train_val)
    pred_test = gb2.predict(X_test)
    print(confusion_matrix(y_train_val, pred_train_val))
    print(confusion_matrix(y_test, pred_test))
    
    return gb2

    
def run_video(activity, video_file, gb):
    if not os.path.isdir(OUT_DIR):
        os.makedirs(OUT_DIR)

    fps = 25
    
    counter_init = 1
    output_prefix = activity

    # Get frame list for each video
    frames = open("00sequences.txt")
    frame_list = frames.read()
    frames.close()
    frame_list = frame_list.split('\n')
    video_frames = {}
    for i in frame_list:
        if i!='':
            video = i.split('\t')
            frame_range = video[-1].split(',')
            n = len(frame_range)
            frame_cuts =  [frame_range[j].strip().split('-') for j in range(n)]
            frame_index = []
            for j in range(1):
                frame_index=frame_index + list(range(int(frame_cuts[j][0]), int(frame_cuts[j][1])+1))
            video_frames[video[0].strip()+'_uncomp.avi'] = frame_index

    # Activity list and params
    activity_list = ['walking', 'jogging', 'running', 'boxing', 'handwaving', 'handclapping']
    params = {'walking': (5, 4, 30), 'jogging': (5, 5, 3), 'running': (5, 6, 1),\
              'boxing': (3, 6, 120), 'handwaving': (5, 5, 30), 'handclapping': (5, 5, 2)}
    
    
    theta, k_size, tao = params[activity]
    X = get_data(activity, video_file, video_frames[video_file], theta, k_size, tao)
    n = X.shape[0]

    # Todo: Complete this part on your own.'
    video = os.path.join(VID_DIR, activity, video_file)
    image_gen = video_frame_generator(video)
    image1 = image_gen.__next__()
    image2 = image_gen.__next__()
    h, w, d = image1.shape
    
    out_path = "output/MHI_{}.mp4".format(video_file)
    video_out = mp4_video_writer(out_path, (w, h), fps)

    output_counter = counter_init

    frame_num = 1
    interpolation = cv2.INTER_CUBIC  # You may try different values
    border_mode = cv2.BORDER_REFLECT101  # You may try different values    
    font = cv2.FONT_HERSHEY_SIMPLEX
    color = (255, 0, 0)
    frame_ids = [10, 30, 50]
    n = X.shape[0]

    while image2 is not None and frame_num<n:

        frame_id = frame_ids[(output_counter - 1) % 3]
            
        if ~np.isnan(X[frame_num,:]).any() and frame_num in video_frames[video_file]:
            action = gb.predict(X[frame_num,:].reshape(1,-1))[0]
            cv2.putText(image2, activity_list[action], (40, 30), font, 0.5, color, 1)

        if frame_num == frame_id:
            out_str = video_file + "-{}.png".format(output_counter)
            save_image(out_str, image2)
            output_counter += 1
        
        video_out.write(image2)
        
        image1 = image2.copy()
        image2 = image_gen.__next__()

        frame_num += 1

    video_out.release()
    
    
def save_videos(person, gb):
    for j in ['walking', 'jogging', 'running', 'boxing', 'handwaving', 'handclapping']:
        path = os.path.join(VID_DIR, j)
        filenames = next(os.walk(path))[2]
        filenames = [i for i in filenames if 'person03' in i]
        for k in filenames:
            run_video(j, k, gb)
    
    if not os.path.isdir(OUT_DIR):
        os.makedirs(OUT_DIR)
        
    fps = 25
    out_path = "output/MHI__{}.mp4".format(person)
    video_out = mp4_video_writer(out_path, (160, 120), fps)
    path = os.path.join(OUT_DIR)
    filenames = next(os.walk(path))[2]
    filenames = [i for i in filenames if 'uncomp.avi.mp4' in i]
    
    # Todo: Complete this part on your own.'
    for i in filenames:
        video = os.path.join(OUT_DIR, i)
        image_gen = video_frame_generator(video)
        image1 = image_gen.__next__()
        image2 = image_gen.__next__()
        h, w, d = image1.shape

        while image2 is not None:    
            video_out.write(image2)
            
            image1 = image2.copy()
            image2 = image_gen.__next__()

    video_out.release()
    

if __name__ == '__main__':
    gb = load('gbm_model')
    x, y = test_model(['person03'], gb)
    save_videos('person03', gb)   