import os
import sys
import json

from logger import logger as log
import message
from message import *
from netlink import Netlink


vanilla_action_keys = {
    "vanilla_alpha",
    "vanilla_beta",
    "vanilla_gamma",
    "vanilla_delta",
}


neo_action_keys = {
    "neo_action"
}



def send_scubic_message(data: dict, nl_sock: Netlink, sock_id):
    if "cubic_beta" in data.keys() and "cubic_bic_scale" in data.keys():
        cubic_beta = int(data["cubic_beta"])
        cubic_bic_scale = int(data["cubic_bic_scale"])
        # log.info(
        #     "cubic_beta: {}, cubic_bic_scale: {}".format(cubic_beta, cubic_bic_scale)
        # )
        msg = UpdateMsg()
        msg.add_field(
            UpdateField().create(VOLATILE_CONTROL_REG, CUBIC_BETA_REG, cubic_beta)
        )
        msg.add_field(
            UpdateField().create(
                VOLATILE_CONTROL_REG, CUBIC_BIC_SCALE_REG, cubic_bic_scale
            )
        )
        update_msg = msg.serialize()
        nl_hdr = SpineMsgHeader()
        nl_hdr.create(NL_UPDATE_FIELDS, len(update_msg) + nl_hdr.hdr_len, sock_id)
        nl_sock.send_msg(nl_hdr.serialize() + update_msg)
        # log.info("send control to kernel flow: {}".format(sock_id))


def send_vanilla_message(msg_data: dict, nl_sock: Netlink, sock_id, msg_type=None):
    nl_hdr = SpineMsgHeader()
    if msg_type == None or msg_type == NL_UPDATE_FIELDS:
        for key in vanilla_action_keys:
            if key not in msg_data:
                log.error("no such key: {}".format(key))
                return

        msg = UpdateMsg()
        for key in vanilla_action_keys:
            postfix = key.split("_")[1]
            reg_name = "VANILLA_{}_REG".format(postfix.upper())
            reg = getattr(message, reg_name)
            msg.add_field(
                UpdateField().create(VOLATILE_CONTROL_REG, reg, msg_data[key])
            )

        update_msg = msg.serialize()
        nl_hdr.create(NL_UPDATE_FIELDS, len(update_msg) + nl_hdr.hdr_len, sock_id)
        nl_sock.send_msg(nl_hdr.serialize() + update_msg)
        # log.info("send control to kernel flow: {}".format(sock_id))
    elif msg_type == NL_MEASURE:
        if not "request_id" in msg_data:
            log.error("No request id for MEASURE message")
            return
        msg = MeasureRequestMsg(int(msg_data["request_id"]))
        msg_raw = msg.serialize()
        nl_hdr.create(NL_MEASURE, len(msg_raw) + nl_hdr.hdr_len, sock_id)
        nl_sock.send_msg(nl_hdr.serialize() + msg_raw)
        
        
def send_neo_message(msg_data: dict, nl_sock: Netlink, sock_id, msg_type=None):
    nl_hdr = SpineMsgHeader()
    if msg_type == None or msg_type == NL_UPDATE_FIELDS:
        for key in neo_action_keys:
            if key not in msg_data:
                log.error("no such key: {}".format(key))
                return

        msg = UpdateMsg()
        for key in neo_action_keys:
            postfix = key.split("_")[1]
            reg_name = "NEO_{}_REG".format(postfix.upper())
            reg = getattr(message, reg_name)
            msg.add_field(
                UpdateField().create(VOLATILE_CONTROL_REG, reg, msg_data[key])
            )

        update_msg = msg.serialize()
        nl_hdr.create(NL_UPDATE_FIELDS, len(update_msg) + nl_hdr.hdr_len, sock_id)
        nl_sock.send_msg(nl_hdr.serialize() + update_msg)
        # log.info("send control to kernel flow: {}".format(sock_id))
    elif msg_type == NL_MEASURE:
        msg = MeasureRequestMsg(int(msg_data))
        msg_raw = msg.serialize()
        nl_hdr.create(NL_MEASURE, len(msg_raw) + nl_hdr.hdr_len, sock_id)
        nl_sock.send_msg(nl_hdr.serialize() + msg_raw)
