#!/usr/bin/env python

import copy
import math
import threading
import numpy as np

import rospy
import smach
import smach_ros
import PyKDL as kdl
import message_filters as mf
import tf_conversions as tfc

from nav_msgs.msg import Odometry
from std_msgs.msg import Bool, UInt8, Duration, Float32, String, ColorRGBA
from std_srvs.srv import Empty, EmptyRequest
from geometry_msgs.msg import PoseStamped, Quaternion, Vector3Stamped
from sensor_msgs.msg import JointState, Joy

from volaly_msgs.msg import EmptyAction, EmptyGoal
from volaly_msgs.msg import WaypointsAction, WaypointsGoal
from volaly_msgs.msg import FollowMeAction, FollowMeGoal
from volaly_msgs.msg import MotionRellocContAction, MotionRellocContGoal

from volaly_msgs.msg import WristGesture
from volaly_msgs.srv import SetWorkspaceShape, SetWorkspaceShapeRequest

from metawear_ros.msg import Vibration, VibrationPattern

from drone_arena_msgs.msg import State as FlightState

flight_state_names = {v: k for k,v in FlightState.__dict__.items() if k[0].isupper()}
workspace_names = {v: k for k,v in SetWorkspaceShapeRequest.__dict__.items() if k[0].isupper() and isinstance(v, int)}
workspace_ids   = {k: v for k,v in SetWorkspaceShapeRequest.__dict__.items() if k[0].isupper() and isinstance(v, int)}

class CBStateExt(smach.State):
    def __init__(self, cb, cb_args=[], cb_kwargs={}, outcomes=[], input_keys=[], output_keys=[], io_keys=[]):
        """Create s state from a single function.
        @type outcomes: list of str
        @param outcomes: Custom outcomes for this state.
        @type input_keys: list of str
        @param input_keys: The userdata keys from which this state might read
        at runtime.
        @type output_keys: list of str
        @param output_keys: The userdata keys to which this state might write
        at runtime.
        @type io_keys: list of str
        @param io_keys: The userdata keys to which this state might write or
        from which it might read at runtime.
        """
        smach.State.__init__(self, outcomes, input_keys, output_keys, io_keys)
        self._cb = cb
        self._cb_args = cb_args
        self._cb_kwargs = cb_kwargs

        if smach.util.has_smach_interface(cb):
            self._cb_input_keys = cb.get_registered_input_keys()
            self._cb_output_keys = cb.get_registered_output_keys()
            self._cb_outcomes = cb.get_registered_outcomes()

            self.register_input_keys(self._cb_input_keys)
            self.register_output_keys(self._cb_output_keys)
            self.register_outcomes(self._cb_outcomes)

    def execute(self, ud):
        return self._cb(self, ud, *self._cb_args, **self._cb_kwargs)

class FsmNode():
    def __init__(self):
        self.bracelet_name = rospy.get_param('~bracelet_name', 'mwear')
        button_topic = rospy.get_param('~button_topic', '{}/button'.format(self.bracelet_name))

        self.sub_button = rospy.Subscriber(button_topic, Bool, self.button_cb)
        self.last_button = None
        self.button_pressed = False
        self.button_released = False

        joy_topic = rospy.get_param('~joy_topic', '/drone/joy')
        self.sub_joy = rospy.Subscriber(joy_topic, Joy, self.joy_cb)
        self.joy_msg = None
        tmp_button = rospy.get_param('~joy_workspace_button', 6) # Left Trigger
        if isinstance(tmp_button, int) and tmp_button > 0:
            self.joy_workspace_button = tmp_button
        else:
            # rospy.logfatal()
            raise ValueError('joy_workspace_button must be integer and greater than 0, instead its value is: {}'.format(tmp_button))

        primary_wspace_name = rospy.get_param('~primary_workspace', 'WORKSPACE_XY_PLANE')
        secondary_wspace_name = rospy.get_param('~secondary_workspace', 'WORKSPACE_CYLINDER')
        try:
            self.primary_wspace = workspace_ids[primary_wspace_name]
            self.secondary_wspace = workspace_ids[secondary_wspace_name]
        except KeyError, e:
            rospy.logfatal('Wrong workspace value: {}. Available workspaces: {}'.format(e.message, workspace_ids.keys()))
            raise e

        rospy.loginfo('Workspaces: {} (primary), {} (secondary)'.format(primary_wspace_name, secondary_wspace_name))

        landing_spot = rospy.get_param('landing_spot', {'x': float('nan'), 'y': float('nan'), 'z': float('nan'), 'tolerance': float('nan')})
        self.landing_spot, self.landing_tolerance = self.get_landing_spot(**landing_spot)

        if math.isnan(self.landing_spot.Norm()):
            rospy.logwarn('Landing spot is not specified, allow landing anywhere')

        wps = rospy.get_param('task_waypoints',
            {'frame_id': 'World',
             'waypoints':
                [{'x':  0.6, 'y': -0.6, 'z':  0.0, 'yaw_deg':  -45.0},
                 {'x':  0.6, 'y':  0.6, 'z':  0.0, 'yaw_deg':   45.0},
                 {'x': -0.6, 'y':  0.6, 'z':  0.0, 'yaw_deg':  135.0},
                 {'x': -0.6, 'y': -0.6, 'z':  0.0, 'yaw_deg': -135.0}
                ]
            })

        self.task_waypoints = [self.waypoint_to_pose(wps['frame_id'], **wp) for wp in wps['waypoints']]

        self.waypoints_action_ns = rospy.get_param('~waypoints_action_ns', '/drone/waypoints_action')
        self.followme_action_ns = rospy.get_param('~followme_action_ns', '/drone/followme_action')
        self.takeoff_action_ns = rospy.get_param('~takeoff_action_ns', '/drone/takeoff_action')
        self.land_action_ns = rospy.get_param('~land_action_ns', '/drone/land_action')
        self.feedback_action_ns = rospy.get_param('~feedback_action_ns', '/drone/feedback_action')
        self.reset_odom_action_ns = rospy.get_param('~reset_odom_action_ns', '/drone/reset_odom_action')

        self.reset_odom_flag = rospy.get_param('~reset_odom', True)


        self.is_motion_relloc = rospy.get_param('~is_motion_relloc', True)
        self.relloc_action_ns = rospy.get_param('~relloc_action_ns', '/motion_relloc/relloc_cont_action')

        ### MOCAP relloc and user geometry calibration
        self.mocap_relloc_action_ns = rospy.get_param('~mocap_relloc_action_ns', '/mocap_relloc/relloc_action')

        human_pose_topic = rospy.get_param('~human_pose_topic', '/optitrack/head')
        self.sub_human_pose = rospy.Subscriber(human_pose_topic, PoseStamped, self.human_pose_cb)
        self.human_pose_msg = None

        human_joint_state_topic = rospy.get_param('~human_joint_state_topic', '{}/joint_states'.format(self.bracelet_name))
        self.pub_human_joint_state = rospy.Publisher(human_joint_state_topic, JointState, latch = True, queue_size = 1)
        ##############################################


        robot_odom_topic = rospy.get_param('~robot_odom_topic', '/drone/odom')
        self.sub_odom = rospy.Subscriber(robot_odom_topic, Odometry, self.robot_odom_cb)
        self.robot_current_pose = PoseStamped()

        robot_state_topic = rospy.get_param('~robot_state_topic', '/drone/flight_state')
        self.last_known_flight_state = FlightState.Landed
        self.sub_flight_state = rospy.Subscriber(robot_state_topic, FlightState, self.flight_state_cb)

        vibration_topic = rospy.get_param('~vibration_topic', '{}/vibration2'.format(self.bracelet_name))
        self.pub_vibration = rospy.Publisher(vibration_topic, Duration, queue_size = 1)

        vibration_pattern_topic = rospy.get_param('~vibration_pattern_topic', '{}/vibration_pattern'.format(self.bracelet_name))
        self.pub_vibration_pattern = rospy.Publisher(vibration_pattern_topic, VibrationPattern, queue_size = 1)

        execute_timer_topic = rospy.get_param('~execute_timer_topic', '{}/execute_timer'.format(self.bracelet_name))
        self.pub_execute_timer = rospy.Publisher(execute_timer_topic, UInt8, queue_size = 1)

        tmp_name = rospy.get_param('~set_yaw_origin_service', '{}/set_yaw_origin'.format(self.bracelet_name))
        self.set_yaw_origin_service = rospy.get_namespace() + tmp_name

        rospy.loginfo('Set yaw origin service: ' + self.set_yaw_origin_service)

        tmp_name = rospy.get_param('~set_workspace_shape_service', 'human/set_workspace_shape')
        self.set_workspace_shape_service = rospy.get_namespace() + tmp_name

        pointing_ray_topic = rospy.get_param('~pointing_ray_topic', 'human/pointing_ray')

        self.motion_dev_topic = rospy.get_param('~motion_dev_topic', '~motion_dev')
        self.pub_motion_dev = rospy.Publisher(self.motion_dev_topic, Float32, queue_size = 100)

        led_feedback_topic = rospy.get_param('~led_topic', '/drone/led')
        self.pub_led_feedback = rospy.Publisher(led_feedback_topic, ColorRGBA, queue_size = 1)

        self.color_blank = ColorRGBA(**{'r': 0.0, 'g': 0.0, 'b': 0.0, 'a': 1.0})
        self.color_auto = ColorRGBA(**{'r': 0.0, 'g': 1.0, 'b': 1.0, 'a': 1.0})
        self.color_followme = ColorRGBA(**{'r': 0.0, 'g': 1.0, 'b': 0.0, 'a': 1.0})
        self.color_detach = ColorRGBA(**{'r': 1.0, 'g': 1.0, 'b': 0.0, 'a': 1.0})

        self.hand_alt_topic = rospy.get_param('~hand_alt_topic', '~hand_alt')
        self.pub_hand_alt = rospy.Publisher(self.hand_alt_topic, Float32, queue_size = 100)

        self.state_name_topic = rospy.get_param('~state_name_topic', '~state')
        self.pub_state_name = rospy.Publisher(self.state_name_topic, String, queue_size = 10, latch = True)

        self.motion_start_threshold = rospy.get_param('~motion_start_threshold', 0.002)
        self.motion_stop_threshold = rospy.get_param('~motion_stop_threshold', 0.005)
        self.motion_timewindow = rospy.get_param('~motion_timewindow', 3.0)
        self.cache_size = self.motion_timewindow * 50; # 3 seconds at 50 Hz

        self.fsub_pointing_ray = mf.Subscriber(pointing_ray_topic, PoseStamped)
        self.cache_pointing_ray = mf.Cache(self.fsub_pointing_ray, self.cache_size)
        self.cache_pointing_ray.registerCallback(self.pointing_ray_cb)

        self.mean_pose = PoseStamped()
        self.max_dev = float('inf')

        self.pointing_rays = []

        hand_altitude_topic = rospy.get_param('~hand_altitude_topic', '{}/altitude'.format(self.bracelet_name))
        self.fsub_hand_altitude = mf.Subscriber(hand_altitude_topic, Vector3Stamped)
        self.cache_hand_altitude = mf.Cache(self.fsub_hand_altitude, 120 * 3.0) # 3 s at 120 Hz
        self.cache_hand_altitude.registerCallback(self.hand_altitude_cb)

        self.mean_hand_alt = float('nan')
        self.max_hand_alt_dev = float('nan')
        self.cur_alt = float('nan')
        self.hand_alts = []

        self.waypoints_goal = WaypointsGoal()
        self.waypoints_goal.loop = True
        self.waypoints_goal.distance_threshold = 0.30 # 20cm
        self.waypoints_goal.yaw_threshold = math.radians(15.0) # 1deg
        self.waypoints_goal.waypoints = self.task_waypoints

        self.followme_goal = FollowMeGoal()
        self.followme_goal.topic = 'human/arm_pointer'
        # self.followme_goal.ignore_z = True
        # self.followme_goal.override_z = 0.6

        self.wrist_gest_topic = rospy.get_param('~wrist_gest_topic', 'human/wrist_gesture')


        self.set_workspace_shape = rospy.ServiceProxy(self.set_workspace_shape_service, SetWorkspaceShape)


        self.sm = smach.StateMachine(outcomes = ['FINISH'])

        self.pub_led_feedback.publish(self.color_blank)

        with self.sm:
            wait_user_sm = smach.Concurrence(outcomes = ['button', 'gesture', 'preempted'],
                                            default_outcome = 'preempted',
                                            outcome_map = {'button': {'CHECK_BUTTON': 'pressed'},
                                                           'gesture': {'CHECK_GESTURE': 'floor_touched'},
                                                           'preempted': {'CHECK_BUTTON':  'preempted',
                                                                         'CHECK_GESTURE': 'preempted'}},
                                            child_termination_cb = lambda _: True
                                          )
            with wait_user_sm:
                smach.Concurrence.add('CHECK_BUTTON', CBStateExt(self.check_button, cb_kwargs = {'context': self}))
                smach.Concurrence.add('CHECK_GESTURE', CBStateExt(self.check_gesture, cb_kwargs = {'context': self}))

            smach.StateMachine.add('WAIT_USER', wait_user_sm,
                transitions = {'button':  'CHECK_DRONE',
                               'gesture': 'LAND',
                               'preempted': 'WAIT_USER'})

            smach.StateMachine.add('CHECK_DRONE', CBStateExt(self.check_drone, cb_kwargs = {'context': self}),
                            transitions = {'Landed':    'RESET_ODOM' if self.reset_odom_flag else 'TAKEOFF',
                                           'Hovering':  'RESET_ODOM_HOVER' if self.reset_odom_flag else 'PERFORM_RELLOC',
                                           'Flying':    'PERFORM_RELLOC',
                                           'TakingOff': 'WAIT_USER',
                                           'Landing':   'WAIT_USER',
                                           'preempted': 'WAIT_USER'})

            smach.StateMachine.add('RESET_ODOM',
                    smach_ros.SimpleActionState(self.reset_odom_action_ns, EmptyAction),
                    transitions = {'succeeded': 'TAKEOFF',
                                   'preempted': 'WAIT_USER',
                                   'aborted':   'WAIT_USER'}
            )

            smach.StateMachine.add('RESET_ODOM_HOVER',
                    smach_ros.SimpleActionState(self.reset_odom_action_ns, EmptyAction),
                    transitions = {'succeeded': 'PERFORM_RELLOC',
                                   'preempted': 'WAIT_USER',
                                   'aborted':   'WAIT_USER'}
            )

            smach.StateMachine.add('TAKEOFF',
                    smach_ros.SimpleActionState(self.takeoff_action_ns, EmptyAction),
                    transitions = {'succeeded': 'WAIT_USER',
                                   'preempted': 'WAIT_USER',
                                   'aborted':   'WAIT_USER'}
            )


            relloc_sm = self.create_relloc_sm(self.is_motion_relloc)

            smach.StateMachine.add('PERFORM_RELLOC', relloc_sm,
                transitions = {'succeeded': 'NOTIFY_SELECTED',
                               'preempted': 'WAIT_USER',
                               'aborted':   'WAIT_USER'})

            notify_selected_sm = smach.Concurrence(outcomes = ['succeeded', 'preempted', 'aborted'],
                                                   default_outcome = 'preempted',
                                                   outcome_map = {'succeeded': {'ROBOT_NOTIFY_SELECTED': 'succeeded'},
                                                                  'preempted': {'ROBOT_NOTIFY_SELECTED': 'preempted'},
                                                                  'aborted':   {'ROBOT_NOTIFY_SELECTED': 'aborted'}}
            )
            with notify_selected_sm:
                smach.Concurrence.add('ROBOT_NOTIFY_SELECTED',
                    smach_ros.SimpleActionState(self.feedback_action_ns, EmptyAction)
                )
                smach.Concurrence.add('USER_NOTIFY_SELECTED',
                    CBStateExt(self.vibrate, cb_kwargs = {'context': self, 'duration': 0.200})
                )

            smach.StateMachine.add('NOTIFY_SELECTED', notify_selected_sm,
                                    transitions = {'succeeded': 'FOLLOW_POINTING',
                                                   'preempted': 'WAIT_USER',
                                                   'aborted': 'WAIT_USER'})


            follow_pointing_sm = smach.Concurrence(outcomes = ['landing', 'detaching', 'aborted'],
                                                   default_outcome = 'detaching',
                                                   outcome_map = {'landing':   {'CHECK_MAX_DEV': 'land',
                                                                                'FOLLOW_ME': 'preempted'},
                                                                  'detaching': {'CHECK_MAX_DEV': 'detach',
                                                                                'FOLLOW_ME': 'preempted'},
                                                                  'aborted':   {'FOLLOW_ME': 'aborted',
                                                                                'WAIT_DRONE': 'Landed'}},
                                                   child_termination_cb = lambda _: True)

            with follow_pointing_sm:
                smach.Concurrence.add('FOLLOW_ME',
                    smach_ros.SimpleActionState(self.followme_action_ns, FollowMeAction, goal = self.followme_goal)
                )

                smach.Concurrence.add('CHECK_MAX_DEV',
                    CBStateExt(self.check_max_deviation, cb_kwargs = {'context': self})
                )

                # smach.Concurrence.add('CHECK_WRIST_GESTURE',
                #     CBStateExt(self.check_wrist_gesture, cb_kwargs = {'context': self})
                # )

                smach.Concurrence.add('ADJUST_WORKSPACE_SHAPE',
                    CBStateExt(self.monitor_joy,
                        cb_kwargs = {'context': self,
                                     'button':  self.joy_workspace_button, # Left Trigger
                                     'released_wspace': self.primary_wspace,
                                     'pressed_wspace':  self.secondary_wspace
                                     })
                )

                smach.Concurrence.add('WAIT_DRONE',
                    CBStateExt(self.wait_drone, cb_kwargs = {'context': self, 'wait_states': ['Landed']})
                )

            smach.StateMachine.add('FOLLOW_POINTING', follow_pointing_sm,
                transitions = {'landing': 'LAND',
                               'detaching': 'USER_NOTIFY_DETACHED',
                               'aborted': 'WAIT_USER'})

            smach.StateMachine.add('LAND',
                    smach_ros.SimpleActionState(self.land_action_ns, EmptyAction),
                    transitions = {'succeeded': 'WAIT_USER',
                                   'preempted': 'WAIT_USER',
                                   'aborted':   'WAIT_USER'}
            )

            smach.StateMachine.add('USER_NOTIFY_DETACHED',
                                    CBStateExt(self.vibrate, cb_kwargs = {'context': self, 'duration': 1.00}),
                                    transitions = {'done': 'WAIT_USER'})

        self.sm.register_start_cb(self.state_transition_cb, cb_args = [self.sm])
        self.sm.register_transition_cb(self.state_transition_cb, cb_args = [self.sm])
        self.sis = smach_ros.IntrospectionServer('smach_server', self.sm, '/SM_RELLOC')

        rospy.wait_for_service(self.set_workspace_shape_service)

    def create_relloc_sm(self, is_motion_relloc = True):
        if is_motion_relloc:
            rospy.loginfo('Creating SM for motion relloc')

            relloc_sm = smach.Concurrence(outcomes = ['succeeded', 'preempted', 'aborted'],
                                          default_outcome = 'preempted',
                                          outcome_map = {'succeeded': {'MOTION_RELLOC': 'succeeded'},
                                                         'preempted': {'MOTION_RELLOC': 'preempted'},
                                                         'aborted':   {'ROBOT_TASK':    'aborted',
                                                                       'MOTION_RELLOC': 'aborted'}},
                                          child_termination_cb = lambda _: True)

            with relloc_sm:
                smach.Concurrence.add('ROBOT_TASK',
                    smach_ros.SimpleActionState(self.waypoints_action_ns, WaypointsAction, goal = self.waypoints_goal)
                )
                smach.Concurrence.add('MOTION_RELLOC',
                    smach_ros.SimpleActionState(self.relloc_action_ns, MotionRellocContAction)
                )

        # else MOCAP relloc
        else:
            rospy.loginfo('Creating SM for MOCAP relloc')

            relloc_sm = smach.Sequence(outcomes = ['succeeded', 'preempted', 'aborted'],
                                        connector_outcome = 'succeeded')

            v_msg = VibrationPattern()
            v_msg.pattern.append(Vibration(power=100, duration=rospy.Duration(0.080)))
            v_msg.pattern.append(Vibration(power=0,   duration=rospy.Duration(0.200)))
            v_msg.pattern.append(Vibration(power=100, duration=rospy.Duration(0.080)))
            v_msg.pattern.append(Vibration(power=0,   duration=rospy.Duration(0.200)))
            v_msg.pattern.append(Vibration(power=100, duration=rospy.Duration(0.080)))

            with relloc_sm:
                # smach.Sequence.add_auto('NOTIFY_USER_RELLOC',
                #     CBStateExt(self.vibrate_pattern, cb_kwargs = {'context': self, 'pattern': v_msg}),
                #     ['done']
                # )
                smach.Sequence.add_auto('NOTIFY_USER_RELLOC',
                    CBStateExt(self.vibrate_3_times, cb_kwargs = {'context': self, 'timer_id': 0}),
                    ['done']
                )
                smach.Sequence.add_auto('CHECK_MAX_DEV',
                    CBStateExt(self.check_max_deviation, cb_kwargs = {'context': self}),
                    ['detach', 'land']
                )
                smach.Sequence.add('ADJUST_USER_GEOM',
                    CBStateExt(self.adjust_user_geom, cb_kwargs = {'context': self}),
                )
                smach.Sequence.add('RESET_POINTING_YAW',
                    smach_ros.ServiceState(self.set_yaw_origin_service, Empty)
                )
                smach.Sequence.add('MOCAP_RELLOC',
                    smach_ros.SimpleActionState(self.mocap_relloc_action_ns, EmptyAction)
                )
                smach.Sequence.add('SLEEP_A_BIT',
                    CBStateExt(self.delay, cb_kwargs = {'context': self, 'duration': 0.5}),
                )
                smach.Sequence.add('SET_DEFAULT_WSPACE',
                    smach_ros.ServiceState(self.set_workspace_shape_service, SetWorkspaceShape,
                                            request = SetWorkspaceShapeRequest('', self.primary_wspace, False)
                    )
                )

        return relloc_sm

    def joy_cb(self, msg):
        self.joy_msg = msg

    def human_pose_cb(self, msg):
        self.human_pose_msg = msg

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

        if states[0] != 'FOLLOW_POINTING':
            self.pub_led_feedback.publish(self.color_auto)

        # rospy.logwarn(args[2].get_active_states())

    def button_cb(self, msg):
        self.button_pressed = False
        self.button_released = False

        if self.last_button != None:
            self.button_pressed = (not self.last_button and msg.data)
            self.button_released = (self.last_button and not msg.data)

        self.last_button = msg.data

    def robot_odom_cb(self, msg):
        # E.g. /drone/odom in 'odom' frame
        self.robot_current_pose = PoseStamped(header = msg.header, pose = msg.pose.pose)

    def pointing_ray_cb(self, msg):
        pointing_rays = self.cache_pointing_ray.getInterval(
                        self.cache_pointing_ray.getLastestTime() - rospy.Duration(0.6),
                        self.cache_pointing_ray.getLastestTime()
                    )

        if pointing_rays:
            # Calculate motion deviation
            self.mean_pose, self.max_dev = self.mean_arm_pose(pointing_rays)
            self.pub_motion_dev.publish(Float32(self.max_dev))
            # rospy.loginfo('Arm poses in cache: {}'.format(len(pointing_rays)))
            # rospy.loginfo('Max Dev: {}'.format(max_dev))

    def flight_state_cb(self, msg):
        self.last_known_flight_state = msg.state

    def hand_altitude_cb(self, msg):
        self.cur_alt = msg.vector.z

        hand_alts = self.cache_hand_altitude.getInterval(
                        self.cache_hand_altitude.getLastestTime() - rospy.Duration(3.0),
                        self.cache_hand_altitude.getLastestTime()
                    )

        if hand_alts:
            # Calculate motion deviation
            self.mean_hand_alt, self.max_hand_alt_dev = self.mean_hand_altitude(hand_alts)
            self.pub_hand_alt.publish(self.mean_hand_alt)
            # self.pub_motion_dev.publish(Float32(self.max_dev))
            # rospy.loginfo('Arm poses in cache: {}'.format(len(pointing_rays)))
            # rospy.loginfo('Max Dev: {}'.format(max_dev))

    def scale_color(self, color, factor):
        c = np.array([color.r, color.g, color.b, color.a]) * factor
        return ColorRGBA(*c)

    @smach.cb_interface(outcomes = ['Landed', 'TakingOff', 'Hovering', 'Flying', 'Landing', 'preempted'])
    def wait_drone(state, udata, context, wait_states = []):
        loop_rate = rospy.Rate(10.0) # 10Hz

        if len(wait_states) == 0:
            # Don't wait, just return the current state
            return flight_state_names[context.last_known_flight_state]

        while not rospy.is_shutdown():
            if flight_state_names[context.last_known_flight_state] in wait_states:
                return flight_state_names[context.last_known_flight_state]

            if state.preempt_requested():
                state.service_preempt()
                break

            loop_rate.sleep()

        return 'preempted'

    @smach.cb_interface(outcomes = ['Landed', 'TakingOff', 'Hovering', 'Flying', 'Landing', 'preempted'])
    def check_drone(state, udata, context, wait_states = []):
        loop_rate = rospy.Rate(10.0) # 10Hz

        if state.preempt_requested():
            state.service_preempt()
            return 'preempted'

        return flight_state_names[context.last_known_flight_state]

    @smach.cb_interface(outcomes = ['pressed', 'preempted'])
    def check_button(state, udata, context):
        loop_rate = rospy.Rate(50.0) # 50Hz

        while not rospy.is_shutdown():
            if context.button_pressed:
                return 'pressed'

            if state.preempt_requested():
                state.service_preempt()
                break

            # if self.button_released:
            #     return 'released'

            loop_rate.sleep()

        return 'preempted'

    @smach.cb_interface(outcomes = ['floor_touched', 'preempted'])
    def check_gesture(state, udata, context):
        loop_rate = rospy.Rate(50.0) # 50Hz

        while not rospy.is_shutdown():
            if context.cur_alt < context.mean_hand_alt - 0.60:
                rospy.logwarn('max_hand_alt_dev < -0.60')
                return 'floor_touched'

            if state.preempt_requested():
                state.service_preempt()
                break

            loop_rate.sleep()

        return 'preempted'

    @smach.cb_interface(outcomes = ['preempted'])
    def check_wrist_gesture(state, udata, context):
        loop_rate = rospy.Rate(20.0) # 20Hz

        def wrist_cb(msg):
            try:
                if msg.gesture == WristGesture.TURN_KEY_CW:
                    context.set_workspace_shape('', SetWorkspaceShapeRequest.WORKSPACE_VISUAL_PLANE)
                elif msg.gesture == WristGesture.TURN_KEY_CCW:
                    context.set_workspace_shape('', SetWorkspaceShapeRequest.WORKSPACE_XY_PLANE)
                else:
                    rospy.logwarn('Unknown wrist gesture. Ignoring')

            except rospy.ServiceException, e:
                rospy.logwarn('Cannot switch workspace shape: {}'.format(e))

        wrist_sub = rospy.Subscriber(context.wrist_gest_topic, WristGesture, wrist_cb)

        while not rospy.is_shutdown():
            if state.preempt_requested():
                state.service_preempt()
                break

            loop_rate.sleep()

        wrist_sub.unregister()

        return 'preempted'

    @smach.cb_interface(outcomes = ['preempted', 'aborted'])
    def monitor_joy(state, udata, context, button, released_wspace, pressed_wspace):
        loop_rate = rospy.Rate(50.0) # 50Hz

        last_joy_buttons = None

        while not rospy.is_shutdown():
            if context.joy_msg and last_joy_buttons:
                # (cur ^ old) -- changed buttons
                changed_buttons = np.logical_xor(context.joy_msg.buttons, last_joy_buttons)

                if context.joy_msg.header.stamp + rospy.Duration(1.0) < rospy.Time.now():
                    context.joy_msg = None
                else:
                    if button < len(context.joy_msg.buttons):
                        try:
                            # Only process if changed
                            if changed_buttons[button]:
                                if context.joy_msg.buttons[button]:
                                    context.set_workspace_shape('', pressed_wspace, False)
                                    rospy.loginfo('Changed workspace to: {}'.format(pressed_wspace))
                                else:
                                    context.set_workspace_shape('', released_wspace, False)
                                    rospy.loginfo('Changed workspace to: {}'.format(released_wspace))

                                context.pub_vibration.publish(rospy.Duration(0.1))
                        except rospy.ServiceException, e:
                            context.pub_led_feedback.publish(ColorRGBA(1.0, 0.0, 0.0, 1.0))
                            rospy.sleep(1.0)
                            context.pub_led_feedback.publish(context.color_auto)
                            rospy.logwarn(e.message)
                    else:
                        rospy.logerr('Specified joy button index [{}] is out of bounds, max index is {}'.format(button, len(context.joy_msg.buttons)))
                        return 'aborted'

            if state.preempt_requested():
                state.service_preempt()
                break

            last_joy_buttons = context.joy_msg.buttons

            loop_rate.sleep()

        return 'preempted'

    @smach.cb_interface(outcomes = ['done'])
    def vibrate(state, udata, context, duration):
        context.pub_vibration.publish(rospy.Duration(duration))
        return 'done'

    @smach.cb_interface(outcomes = ['done'])
    def vibrate_pattern(state, udata, context, pattern):
        context.pub_vibration_pattern.publish(pattern)
        return 'done'

    @smach.cb_interface(outcomes = ['done'])
    def vibrate_3_times(state, udata, context, timer_id):
        context.pub_execute_timer.publish(timer_id)
        return 'done'

    @smach.cb_interface(outcomes = ['land', 'detach', 'preempted'])
    def check_max_deviation(state, udata, context):
        loop_rate = rospy.Rate(10.0) # 10Hz

        goal_time = rospy.Time.now() + rospy.Duration(3.0) # fly at least 3s

        hold_dur = rospy.Duration(1.0)
        hold_counter_init = 3
        context.hold_counter = hold_counter_init # hold arm still for 3s
        context.hold_flag = False

        hold_timer = None

        time_steps = (hold_dur * hold_counter_init) / loop_rate.sleep_dur
        dim_decrement = 1.0 / time_steps
        dim_factor = 1.0
        dim_color = context.scale_color(context.color_detach, dim_factor)

        blink_counter = 0

        context.pub_led_feedback.publish(context.color_followme)

        while not rospy.is_shutdown():
            blink_counter += 1

            if not hold_timer:
                if blink_counter % 5 == 0:
                    context.pub_led_feedback.publish(context.color_followme)
                else:
                    context.pub_led_feedback.publish(context.scale_color(context.color_followme, 0.3))

            if rospy.Time.now() > goal_time:
                if not hold_timer:
                    if context.max_dev < context.motion_stop_threshold:
                        rospy.loginfo('Hold timer is on')

                        def hold_cb(e):
                            context.hold_counter -= 1
                            if context.hold_counter > 0:
                                context.pub_vibration.publish(rospy.Duration(0.150))
                            else:
                                context.hold_flag = True

                        hold_timer = rospy.Timer(hold_dur, hold_cb)
                else:
                    context.pub_led_feedback.publish(dim_color)

                    dim_factor -= dim_decrement
                    dim_factor = 1.0 if dim_factor < 0.0 else dim_factor
                    dim_color = context.scale_color(context.color_detach, dim_factor)

                    if context.hold_flag:
                        hold_timer.shutdown()
                        hold_timer = None

                        # if context.is_at_landing_spot():
                        #     return 'land'
                        # else:
                        #     return 'detach'

                        context.pub_led_feedback.publish(context.color_auto)
                        return 'detach'

                    if context.max_dev > context.motion_start_threshold:
                        hold_timer.shutdown()
                        hold_timer = None
                        context.hold_counter = hold_counter_init

                        dim_factor = 1.0
                        blink_counter = 0
                        context.pub_led_feedback.publish(context.color_followme)

                        rospy.loginfo('Hold timer is canceled')

            if state.preempt_requested():
                state.service_preempt()
                break

            loop_rate.sleep()

        context.pub_led_feedback.publish(context.color_auto)
        return 'preempted'

    @smach.cb_interface(outcomes = ['succeeded', 'preempted', 'aborted'])
    def adjust_user_geom(state, udata, context):
        if not context.human_pose_msg:
            rospy.logerr('User\'s MOCAP pose is not known. Aborting...')
            return 'aborted'

        if state.preempt_requested():
            state.service_preempt()
            return 'preempted'

        scale_factor = context.human_pose_msg.pose.position.z / 1.83

        j_state = JointState()
        j_state.header.stamp = context.human_pose_msg.header.stamp
        j_state.name = ['footprint_to_neck', 'shoulder_to_wrist', 'wrist_to_finger']
        j_state.position = [1.47 * scale_factor, 0.51 * scale_factor, 0.18 * scale_factor]
        context.pub_human_joint_state.publish(j_state)

        return 'succeeded'

    @smach.cb_interface(outcomes = ['succeeded', 'preempted', 'aborted'])
    def delay(state, udata, context, duration):
        loop_rate = rospy.Rate(50) # 50Hz

        timer = None
        timer_flag = {'value': False} # mutable object

        while not rospy.is_shutdown():
            if state.preempt_requested():
                state.service_preempt()
                if timer:
                    timer.shutdown()
                break

            if not timer:
                def timer_cb(e):
                    timer_flag['value'] = True
                    rospy.loginfo('Delay timer fired')

                try:
                    timer = rospy.Timer(rospy.Duration(duration), timer_cb, oneshot = True)
                except e:
                    rospy.logerr(e)
                    return 'aborted'

            if timer_flag['value']:
                return 'succeeded'

            loop_rate.sleep()

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

    def mean_arm_pose(self, pointing_rays):
        tmp = kdl.Vector(0.0, 0.0, 0.0)
        direction = kdl.Vector()

        for m in pointing_rays:
            p = tfc.fromMsg(m.pose)
            tmp = p.M * kdl.Vector(1.0, 0.0, 0.0)
            direction = direction + tmp

        n = direction.Normalize()
        # rospy.loginfo('x: {} y: {} z: {}'.format(direction.x(), direction.y(), direction.z()))

        pitch = math.atan2(-direction.z(), math.sqrt(direction.x()*direction.x() + direction.y()*direction.y()))
        yaw = math.atan2(direction.y(), direction.x())

        pose = kdl.Frame(kdl.Rotation.RPY(0.0, pitch, yaw))

        pointing_ray_msg = copy.deepcopy(pointing_rays[-1])
        pointing_ray_msg.pose = tfc.toMsg(pose)

        max_dev = 0.0
        for m in pointing_rays:
            p = tfc.fromMsg(m.pose)
            tmp = p.M * kdl.Vector(1.0, 0.0, 0.0)
            dev = (direction - tmp).Norm()
            if dev > max_dev: max_dev = dev

        return pointing_ray_msg, max_dev

    def mean_hand_altitude(self, hand_alts):
        tmp = 0.0

        lst = []
        for m in hand_alts:
            tmp = tmp + m.vector.z
            # lst.append(m.vector.z)

        mean_val = tmp / len(hand_alts)

        # print(lst)

        max_dev = 0.0
        for m in hand_alts:
            dev = m.vector.z - mean_val
            max_dev = min(max_dev, dev)

            # if math.fabs(dev) > math.fabs(max_dev): max_dev = dev
            # if math.fabs(dev) > math.fabs(max_dev): max_dev = dev

        # rospy.loginfo_throttle(1.0, 'mean alt: {}, max_dev alt: {}'.format(mean_val, max_dev))
        # rospy.loginfo('mean alt: {}, max_dev alt: {}'.format(mean_val, max_dev))

        # print(mean_val, max_dev)
        return mean_val, max_dev

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
