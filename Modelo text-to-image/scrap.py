from PIL import Image
import json
import numpy as np

dataset_names = ["cartoon", 
          "digital_art",
          "kids_drawing", 
          "oil_painting",
          "sketch",
          "stained_glass",
          "watercolor"]
rutaJson = "./Humanart/jsons/"
h = 64

clases = {"cartoon":        [1.,0.,0.,0.,0.,0.,0.], 
          "digital_art":    [0.,1.,0.,0.,0.,0.,0.],
          "kids_drawing":   [0.,0.,1.,0.,0.,0.,0.], 
          "oil_painting":   [0.,0.,0.,1.,0.,0.,0.],
          "sketch":         [0.,0.,0.,0.,1.,0.,0.],
          "stained_glass":  [0.,0.,0.,0.,0.,1.,0.],
          "watercolor":     [0.,0.,0.,0.,0.,0.,1.]}

def get_img(img_name):
    img = Image.open(f"./{img_name}").resize(size=(h, h))
    img = img.convert('RGB')
    return img

if __name__ == "__main__":
    dataset = dict()
    dataset['name'] = 'humanArt'
    dataset['shape'] = (3, h, h)
    data = dict()
    count = 0
    for dataset_name in dataset_names:

        print('Iniciando scrap de '+dataset_name)

        json_file = rutaJson+'training_humanart_'+dataset_name+'.json'
        with open(json_file) as lista:
            dato = json.load(lista)
            for idx, imagen in enumerate(dato['images']):
                img = get_img(imagen['file_name'])
                #desc = imagen['description']
                clase = clases[imagen['category'].replace(' ', '_')]
                
                data[count]={'img':img,'label':clase}
                count += 1
                if idx % 200 == 0:
                    print(f'------ se han guardado {idx} img de training ------')

        json_file = rutaJson+'validation_humanart_'+dataset_name+'.json'
        with open(json_file) as lista:
            dato = json.load(lista)
            for idx, imagen in enumerate(dato['images']):
                img = get_img(imagen['file_name'])
                #desc = imagen['description']
                clase = clases[imagen['category'].replace(' ', '_')]
                
                data[count]={'img':img,'label':clase}
                count += 1
                if idx % 200 == 0:
                    print(f'------ se han guardado {idx} img de validation ------')


    dataset['dataset'] = data
    dataset['length'] = len(data)
    np.save(f'./datasets/humanart_clases.npy', dataset)