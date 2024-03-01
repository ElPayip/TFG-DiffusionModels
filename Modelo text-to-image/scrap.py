from PIL import Image
import json
import numpy as np

dataset_name = 'kids_drawing'
rutaJson = "./Humanart/jsons/"
h = 256

def get_img(img_name):
    img = Image.open(f"./{img_name}").resize(size=(h, h))
    img = img.convert('RGB')
    return img

if __name__ == "__main__":
    dataset = dict()
    dataset['name'] = 'humanArt_'+dataset_name
    dataset['shape'] = (3, h, h)
    data = dict()
    count = 0

    print('Iniciando scrap de '+dataset_name)

    json_file = rutaJson+'training_humanart_'+dataset_name+'.json'
    with open(json_file) as lista:
        dato = json.load(lista)
        for idx, imagen in enumerate(dato['images']):
            img = get_img(imagen['file_name'])
            desc = imagen['description']
            
            data[count]={'img':img,'label':desc}
            count += 1
            if idx % 200 == 0:
                print(f'------ se han guardado {idx} img de training ------')

    json_file = rutaJson+'validation_humanart_'+dataset_name+'.json'
    with open(json_file) as lista:
        dato = json.load(lista)
        for idx, imagen in enumerate(dato['images']):
            img = get_img(imagen['file_name'])
            desc = imagen['description']
            
            data[count]={'img':img,'label':desc}
            count += 1
            if idx % 200 == 0:
                print(f'------ se han guardado {idx} img de validacion ------')


    dataset['dataset'] = data
    dataset['length'] = len(data)
    np.save(f'./datasets/humanart_'+dataset_name+'_256.npy', dataset)