import os
import boto3
import time
from multiprocessing import Process, Pipe

model_slices = [
    'tmp3_s0.onnx', 'tmp3_s1.onnx',
    'tmp3_s2.onnx'
]

s3 = boto3.resource('s3')
bucket_name = 'inference-single-lambda-bucketformodelanddata-1tbmd4frldeta'
bucket = s3.Bucket(bucket_name)

sesses = {}
labels = []
logs = []
sess_processes = {}
send_conn_parent, send_conn_child = None, None
    
start_milli_time = int(time.time() * 1000)

def lambda_handler(event, context):
    def download_slice(slice_name):
        filename = '/tmp/' + slice_name
        if not os.path.isfile(filename):
            bucket.download_file('model/' + slice_name, filename)
    
    
    def download_process(model_slices, conn):
        for model_slice in model_slices:
            timer_func(download_slice, model_slice)
            conn.send(model_slice)
        print(logs)
        conn.send("End")
    
    
    def timer_func(func_name, *args):
        global logs
        now_milli_time = int(time.time() * 1000) - start_milli_time
        res = func_name(*(args))
        logs.append([
            func_name.__name__, now_milli_time,
            int(time.time() * 1000) - now_milli_time - start_milli_time
        ])
        return res
    
    
    (conn1, conn2) = Pipe()
    p = Process(target=timer_func, args=(
        download_process,
        model_slices,
        conn1,
    ))
    p.start()
    
    import requests
    import onnxruntime
    import numpy as np
    from PIL import Image
    from io import BytesIO
    import pickle
    
    
    def download_img(url):
        r = requests.get(url)
        img = Image.open(BytesIO(r.content))
        return img
    
    
    def preprocess(img):
        img = img.resize((256, 256))
        img = img.crop((16, 16, 240, 240))
        img = np.array(img).astype(np.float32) / 255.
        img = np.rollaxis(img, 2, 0)
        for channel, mean, std in zip(range(3), [0.485, 0.456, 0.406],
                                      [0.229, 0.224, 0.225]):
            img[channel, :, :] -= mean
            img[channel, :, :] /= std
        img = np.expand_dims(img, axis=0)
        return img
    
    
    def load_slice(slice_name):
        sess_options = onnxruntime.SessionOptions()
        sess = onnxruntime.InferenceSession('/tmp/' + slice_name, sess_options)
        return sess
    
    
    def inference(sess, inp):
        feed = dict([(input.name, inp) for n, input in enumerate(sess.get_inputs())
                    ])
        output = sess.run(None, feed)[0]
        return output
    
    
    def postprocess(idx):
        global labels
        if not labels:
            r = requests.get('https://s3.amazonaws.com/onnx-model-zoo/synset.txt')
            labels = [l.rstrip() for l in r.text.splitlines()]
        return labels[idx]
    
    
    def sess_process(model_slice, root_conn, send_conn, recv_conn):
        root_conn.recv()
        if model_slice != model_slices[0]:
            send_conn.recv()
        sess = timer_func(load_slice, model_slice)
        if model_slice != model_slices[-1]:
            recv_conn.send(model_slice)
        # todo cannot handle second request
        while True:
            inp = pickle.loads(send_conn.recv())
            output = timer_func(inference, sess, inp)
            recv_conn.send(pickle.dumps(output))
            print(logs)
    
    
    def init_process():
        global sess_processes
        send_conn_parent, send_conn_child = Pipe()
        for model_slice in model_slices:
            root_conn_parent, root_conn_child = Pipe()
            recv_conn_parent, recv_conn_child = Pipe()
            p = Process(target=sess_process,
                        args=(
                            model_slice,
                            root_conn_child,
                            send_conn_child,
                            recv_conn_parent,
                        ))
            p.start()
            sess_processes[model_slice] = (p, root_conn_parent)
            send_conn_child = recv_conn_child
        return send_conn_parent, send_conn_child
    
    
    def inference_process(url, conn):
        img = timer_func(download_img, url)
        img = timer_func(preprocess, img)
        global sess_processes, send_conn_parent, send_conn_child
        if not sess_processes:
            send_conn_parent, send_conn_child = init_process()
            while True:
                msg = conn.recv()
                if msg == "End":
                    break
                p, root_conn = sess_processes[msg]
                root_conn.send(msg)
                if msg == model_slices[0]:
                    send_conn_parent.send(pickle.dumps(img))
        else:
            send_conn_parent.send(pickle.dumps(img))
        img = pickle.loads(send_conn_child.recv())
        output = timer_func(postprocess, img.argmax())
        return output
    

    img = timer_func(inference_process, event['img_url'], conn2)
    p.join()
    print(logs)
    return img
