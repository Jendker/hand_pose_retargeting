from pydrake.multibody.parsing import Parser
from pydrake.multibody.plant import AddMultibodyPlantSceneGraph
from pydrake.systems.framework import DiagramBuilder
import numpy as np
import os


class ForwardKinematics:
    def __init__(self):
        builder = DiagramBuilder()
        self.plant, _ = AddMultibodyPlantSceneGraph(builder)
        file_dir = os.path.dirname(__file__)
        self.instance = Parser(self.plant).AddModelFromFile(file_dir + "/model/shadowrobot.urdf")
        self.plant.Finalize()
        self.context = self.plant.CreateDefaultContext()

    def updateForwardKinematics(self, joint_positions):
        self.plant.SetPositions(self.context, self.instance, joint_positions)

    def getLocalBodyPosition(self, body_name):
        body = self.plant.GetBodyByName(body_name)
        return np.array(self.plant.EvalBodyPoseInWorld(self.context, body))

    def getWorldBodyPosition(self, transformation_matrix, body_name):
        return (transformation_matrix @ np.append(self.getLocalBodyPosition(body_name), 1))[0:3]
