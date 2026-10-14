"""Explicit tactical heuristics for evaluation, never inserted into the frozen curriculum."""
import numpy as np


class TacticalBot:
    STYLES=('interceptor','counter_defender','dribbler','aimed_striker')

    def __init__(self,style):
        if style not in self.STYLES:raise ValueError(style)
        self.style=style

    def act(self,state,players,n_red=1):
        players=np.asarray(players)
        ball=state.ball_pos[:,None,:]
        pos=state.player_pos[:,players]
        velocity=state.ball_vel[:,None,:]
        own=state.goal_center(False)[:,players]
        goal=state.goal_center(True)[:,players].copy()
        opponent=state.player_pos[:,1-players]
        # Aim away from the opponent within the actual finite goal mouth.
        a,b=state.goal_line(True)
        lo=np.minimum(a[:,players,1],b[:,players,1])+8
        hi=np.maximum(a[:,players,1],b[:,players,1])-8
        goal[...,1]=np.clip(-opponent[...,1],lo,hi)
        direction=goal-ball
        direction/=np.maximum(np.linalg.norm(direction,axis=-1,keepdims=True),1e-9)
        target=ball-direction*22
        delta=ball-pos;distance=np.linalg.norm(delta,axis=-1)
        aligned=np.sum(delta*direction,axis=-1)>distance*.8
        kick=(distance<24.8)&aligned
        if self.style=='interceptor':
            # Linear short-horizon forecast, not a full engine oracle; ignores rebounds.
            horizon=np.clip(distance/max(state.player_max_speed,1e-6),0,48)
            target=ball+velocity*horizon[...,None]
            kick=distance<24.8
        elif self.style=='counter_defender':
            defending=np.sum((ball-own)*(goal-own),axis=-1)<.55*np.sum((goal-own)**2,axis=-1)
            guard=own.copy();guard[...,0]*=.78
            t=np.clip((guard[...,0]-ball[...,0])/np.where(abs(velocity[...,0])>.05,velocity[...,0],1),0,60)
            guard[...,1]=np.clip(ball[...,1]+velocity[...,1]*t,lo,hi)
            target=np.where(defending[...,None]&(distance>65)[...,None],guard,target)
        elif self.style=='dribbler':
            # Carry toward a lane away from the defender, release kick until pressured.
            dodge=np.where(opponent[...,1]>=ball[...,1],-1.,1.)
            lane=direction.copy();lane[...,1]+=.65*dodge
            lane/=np.maximum(np.linalg.norm(lane,axis=-1,keepdims=True),1e-9)
            target=ball-lane*21
            pressured=np.linalg.norm(opponent-ball,axis=-1)<55
            close_goal=np.linalg.norm(goal-ball,axis=-1)<140
            kick=(distance<24.8)&aligned&(pressured|close_goal)
        move=target-pos
        move[...,0]*=np.where(players>=n_red,-1,1)
        x=np.where(move[...,0]>2,2,np.where(move[...,0]<-2,0,1))
        y=np.where(move[...,1]>2,2,np.where(move[...,1]<-2,0,1))
        return np.stack([x,y,kick.astype(int)],axis=-1).astype(np.int64)
