#!/usr/bin/env python

import copy
import math
import threading

import rospy
import rostopic
import smach
import smach_ros
import PyKDL as kdl
import message_filters as mf
import tf_conversions as tfc

from nav_msgs.msg import Odometry
from std_msgs.msg import Bool, Duration, Float32, String
from geometry_msgs.msg import PoseStamped, Quaternion

from volaly_msgs.msg import EmptyAction, EmptyGoal
from volaly_msgs.msg import WaypointsAction, WaypointsGoal
from volaly_msgs.msg import FollowMeAction, FollowMeGoal
from volaly_msgs.msg import MotionRellocContAction, MotionRellocContGoal

from volaly_msgs.msg import WristGesture
from volaly_msgs.srv import SetWorkspaceShape, SetWorkspaceShapeRequest

from metawear_ros.msg import Vibration, VibrationPattern

from drone_arena_msgs.msg import State as FlightState

flight_state_names = {v: k for k,v in FlightState.__dict__.items() if k[0].isupper()}

class FsmNode():
    def __init__(self):
        button_topic = rospy.get_param('~button_topic', '/drone/joy/buttons[4]')
        button_topic_class, button_real_topic, self.button_eval_func = rostopic.get_topic_class(rospy.resolve_name(button_topic))

        self.sub_button = rospy.Subscriber(button_real_topic, button_topic_class, self.button_cb)
        self.last_button = None
        self.button_pressed = False
        self.button_released = False

        self.land_action_ns = rospy.get_param('~land_action_ns', '/drone/land_action')

        robot_odom_topic = rospy.get_param('~robot_odom_topic', '/drone/odom')
        self.sub_odom = rospy.Subscriber(robot_odom_topic, Odometry, self.robot_odom_cb)
        self.robot_current_pose = PoseStamped()

        self.state_name_topic = rospy.get_param('~state_name_topic', '~state')
        self.pub_state_name = rospy.Publisher(self.state_name_topic, String, queue_size = 10, latch = True)

        robot_state_topic = rospy.get_param('~robot_state_topic', '/drone/flight_state')
        self.last_known_flight_state = FlightState.Landed
        self.sub_flight_state = rospy.Subscriber(robot_state_topic, FlightState, self.flight_state_cb)
        self.flight_state_event = threading.Event()

        landing_spot = rospy.get_param('landing_spot', {'x': float('nan'), 'y': float('nan'), 'z': float('nan'), 'tolerance': float('nan')})
        self.landing_spot, self.landing_tolerance = self.get_landing_spot(**landing_spot)

        if math.isnan(self.landing_spot.Norm()):
            rospy.logwarn('Landing spot is not specified, allow landing anywhere')

        self.sm = smach.StateMachine(outcomes = ['FINISH'])

        with self.sm:
            # smach.StateMachine.add('WAIT_USER',
            #     smach.CBState(self.check_button, cb_kwargs = {'context': self}),
            #     transitions = {'pressed': 'FOLLOW_POINTING',
            #                    'preempted': 'FINISH'})

            smach.StateMachine.add('WAIT_USER',
                smach.CBState(self.wait_for_flying, cb_kwargs = {'context': self}),
                transitions = {'succeeded': 'FOLLOW_POINTING',
                               'aborted':   'FINISH',
                               'preempted': 'WAIT_USER'})

            smach.StateMachine.add('FOLLOW_POINTING',
                smach.CBState(self.wait_for_landing, cb_kwargs = {'context': self}),
                transitions = {'succeeded': 'LAND',
                               'aborted':   'FOLLOW_POINTING',
                               'preempted': 'FINISH'})

            smach.StateMachine.add('LAND',
                    smach_ros.SimpleActionState(self.land_action_ns, EmptyAction),
                    transitions = {'succeeded': 'WAIT_USER',
                                   'preempted': 'FINISH',
                                   'aborted':   'WAIT_USER'})

        self.sm.register_start_cb(self.state_transition_cb, cb_args = [self.sm])
        self.sm.register_transition_cb(self.state_transition_cb, cb_args = [self.sm])
        self.sis = smach_ros.IntrospectionServer('smach_server', self.sm, '/SM_JOY')

    def get_landing_spot(self, x, y, z, tolerance = 0.6):
        return kdl.Vector(float(x), float(y), float(z)), tolerance

    def yaw_to_quat(self, yaw):
        q = kdl.Rotation.RPY(0.0, 0.0, yaw).GetQuaternion()
        return Quaternion(*q)

    def waypoint_to_pose(self, frame_id, x, y, z, yaw_deg):
        p = PoseStamped()

        p.header.frame_id = frame_id
        p.pose.position.x = x
        p.pose.position.y = y
        p.pose.position.z = z
        p.pose.orientation = self.yaw_to_quat(math.radians(yaw_deg))

        return p

    def state_transition_cb(self, *args, **kwargs):
        states = args[2].get_active_states()
        s = '-'.join(states)

        self.pub_state_name.publish(String(s))

        # rospy.logwarn(args[2].get_active_states())

    def button_cb(self, msg):
        if self.button_eval_func:
            data = self.button_eval_func(msg)
        else:
            data = copy.deepcopy(msg.data)

        self.button_pressed = False
        self.button_released = False

        if self.last_button != None:
            self.button_pressed = (not self.last_button and data)
            self.button_released = (self.last_button and not data)

        self.last_button = data

    def robot_odom_cb(self, msg):
        # E.g. /drone/odom in 'odom' frame
        self.robot_current_pose = PoseStamped(header = msg.header, pose = msg.pose.pose)

    def flight_state_cb(self, msg):
        self.last_known_flight_state = msg.state
        self.flight_state_event.set()

    @smach.cb_interface(outcomes = ['pressed', 'preempted'])
    def check_button(udata, context):
        loop_rate = rospy.Rate(50.0) # 50Hz

        while not rospy.is_shutdown():
            if context.button_pressed:
                return 'pressed'
            # if self.button_released:
            #     return 'released'
            loop_rate.sleep()

        return 'preempted'

    @smach.cb_interface(outcomes = ['succeeded', 'preempted', 'aborted'])
    def wait_for_landing(udata, context):
        while not rospy.is_shutdown():
            if context.last_known_flight_state == FlightState.Landed:
                if context.is_at_landing_spot():
                    return 'succeeded'
                else:
                    return 'aborted'

            # Sleep at max 0.1s, if there is an event it will wake up earlier
            context.flight_state_event.wait(0.1)
            # Force to wait another event
            context.flight_state_event.clear()

        return 'preempted'

    @smach.cb_interface(outcomes = ['succeeded', 'preempted', 'aborted'])
    def wait_for_flying(udata, context):
        while not rospy.is_shutdown():
            if context.last_known_flight_state in [FlightState.TakingOff, FlightState.Hovering, FlightState.Flying]:
                return 'succeeded'

            # Sleep at max 0.1s, if there is an event it will wake up earlier
            context.flight_state_event.wait(0.1)
            # Force to wait another event
            context.flight_state_event.clear()

        return 'preempted'

    def is_at_landing_spot(self):
        rp = kdl.Vector(self.robot_current_pose.pose.position.x,
                        self.robot_current_pose.pose.position.y,
                        0.0)

        ls = self.landing_spot
        d = (rp - ls).Norm()

        if math.isnan(d):
            rospy.loginfo('Landing anywhere')
            return True

        # rospy.loginfo('Robot pose: {}'.format(rp))
        if d < (self.landing_tolerance * 1.5):
            rospy.loginfo('Distance to landing spot: {}'.format(d))

        if d < self.landing_tolerance:
            return True
        else:
            return False

    def mean_arm_pose(self, arm_poses):
        tmp = kdl.Vector(0.0, 0.0, 0.0)
        elbow_angle = 0.0
        direction = kdl.Vector()

        for m in arm_poses:
            elbow_angle = elbow_angle + m.elbow_angle
            p = tfc.fromMsg(m.direction)
            tmp = p.M * kdl.Vector(1.0, 0.0, 0.0)
            direction = direction + tmp

        n = direction.Normalize()
        # rospy.loginfo('x: {} y: {} z: {}'.format(direction.x(), direction.y(), direction.z()))
        elbow_angle = elbow_angle / len(arm_poses)

        pitch = math.atan2(-direction.z(), math.sqrt(direction.x()*direction.x() + direction.y()*direction.y()))
        yaw = math.atan2(direction.y(), direction.x())

        pose = kdl.Frame(kdl.Rotation.RPY(0.0, pitch, yaw))

        arm_msg = copy.deepcopy(arm_poses[-1])
        arm_msg.direction = tfc.toMsg(pose)
        arm_msg.elbow_angle = elbow_angle

        max_dev = 0.0
        for m in arm_poses:
            p = tfc.fromMsg(m.direction)
            tmp = p.M * kdl.Vector(1.0, 0.0, 0.0)
            dev = (direction - tmp).Norm()
            if dev > max_dev: max_dev = dev

        return arm_msg, max_dev


    def run(self):
        rospy.on_shutdown(lambda: self.sm.request_preempt())

        self.sis.start()

        smach_thread = threading.Thread(target = self.sm.execute)
        smach_thread.start()

        rospy.spin()
        smach_thread.join()

        self.sis.stop()

if __name__ == '__main__':
    rospy.init_node('relloc_fsm_node')

    fsm = FsmNode()

    try:
        fsm.run()
    except rospy.ROSInterruptException:
        rospy.logdebug('Exiting')
        pass
