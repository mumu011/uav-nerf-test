import numpy as np
import json
import os
from os.path import join as opj
from os.path import dirname as opd

global output_dir
output_dir = opj(
    opd(opd(__file__)),
    "output_v3"
)

class NumpyArrayEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def read():
    file_name = opj(
        output_dir,
        "transforms.json"
    )
    with open(file_name, "r") as f:
        content = json.load(f)
    print(content)

def qvec2rotmat(qvec):
	return np.array([
		[
			1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
			2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
			2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]
		], [
			2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
			1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
			2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]
		], [
			2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
			2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
			1 - 2 * qvec[1]**2 - 2 * qvec[2]**2
		]
	])

def create_translation_matrix(offset):
    '''
    Create a transformation matrix that translates a vetor by the given offset
    
    Parameters
    -----------
    offset - np.ndarray, shape - (3,)
        The translation offset
    
    Returns
    ----------
    T - np.ndarray, shape - (4, 4)
        The translation matrix
    '''
    T = np.identity(4)
    T[:3, 3] = offset
    return T

def transform(json_idx: int):
    global output_dir
    # test
    if (json_idx == -1):
        file_name = opj(
            output_dir,
            "test.json"
        )
    else:
        file_name = opj(
            output_dir,
            f"test_{json_idx}.json"
        )
    with open(file_name, "r") as f:
        content = json.load(f)

    imgs_num = len(content["frames"])
    x_list = np.zeros(imgs_num)
    y_list = np.zeros(imgs_num)
    z_list = np.zeros(imgs_num)
    q_w = np.zeros(imgs_num)
    q_x = np.zeros(imgs_num)
    q_y = np.zeros(imgs_num)
    q_z = np.zeros(imgs_num)

    for i in range(imgs_num):
        x_list[i] = content["frames"][i]["position"]["x"]
        y_list[i] = content["frames"][i]["position"]["y"]
        z_list[i] = content["frames"][i]["position"]["z"]
        q_w[i] = content["frames"][i]["quaternion"]["w"]
        q_x[i] = content["frames"][i]["quaternion"]["x"]
        q_y[i] = content["frames"][i]["quaternion"]["y"]
        q_z[i] = content["frames"][i]["quaternion"]["z"]

    # TODO: del repeat
    # print(f"xlist:\n{x_list}")
    # print(f"ylist:\n{y_list}")
    # print(f"zlist:\n{z_list}")

    x_min = np.min(x_list)
    x_max = np.max(x_list)
    y_min = np.min(y_list)
    y_max = np.max(y_list)
    z_min = np.min(z_list)
    z_max = np.max(z_list)

    # ORIGIN = [x_min, y_min, z_min]
    # ORIGIN = [(x_min + x_max) / 2, (y_min + y_max) / 2, (z_min + z_max) / 2]
    ORIGIN = [5, 5, 2]
    # X_LEN = x_max - x_min
    # Y_LEN = y_max - y_min
    # Z_LEN = z_max - z_min
    scale = 40
    x_list = (x_list - ORIGIN[0]) / scale
    y_list = (y_list - ORIGIN[1]) / scale
    z_list = (z_list - ORIGIN[2]) / scale
    # x_list = (x_list - ORIGIN[0]) / X_LEN / 2
    # y_list = (y_list - ORIGIN[1]) / Y_LEN / 2
    # z_list = (z_list - ORIGIN[2]) / Z_LEN / 2

    re = []
    idx = []
    for i in range(imgs_num):
        qvec = np.array([q_w[i], q_x[i], q_y[i], q_z[i]])
        R = qvec2rotmat(qvec)

        R_ = np.identity(4)
        R = R[:, [1,2,0]]
        R_[:3, :3] = R

        offset = np.array([x_list[i], y_list[i], z_list[i]])
        T_ = create_translation_matrix(offset)

        xform = T_ @ R_

        if xform.tolist() in re:
            idx.append(i)
    
        content["frames"][i].pop("position")
        content["frames"][i].pop("quaternion")
        content["frames"][i].update({'transform_matrix': xform})
        re.append(xform.tolist())
    
    for counter, index in enumerate(idx):
        index = index - counter
        content["frames"].pop(index)

    file_name = opj(
        output_dir,
        "transforms.json"
    )
    data_json = json.dumps(content, cls=NumpyArrayEncoder, indent=2)
    with open(file_name, "w", newline='\n') as f:
        f.write(data_json)
        f.close()
        print("done")

if __name__ == "__main__":
    transform(-1)
    # read()