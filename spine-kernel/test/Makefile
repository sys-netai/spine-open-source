
DEBUG = n
ONE_PIPE = n
IPC = 0 # 0 == netlink, 1 == chardev

# Add your debugging flag (or not) to EXTRA_CFLAGS
ifeq ($(DEBUG),y)
  DEBFLAGS = -O1 -g -D__DEBUG__ -D__LOG_DEBUG__ # "-O" is needed to expand inlines
else
  DEBFLAGS = -Ofast
endif

KERNEL_VERSION_MAJOR := $(shell uname -r | awk -F '.' '{print $$1}')
KERNEL_VERSION_MINOR := $(shell uname -r | awk -F '.' '{print $$2}')
EXTRA_CFLAGS += -D__KERNEL_VERSION_MAJOR__=$(KERNEL_VERSION_MAJOR) 
EXTRA_CFLAGS += -D__KERNEL_VERSION_MINOR__=$(KERNEL_VERSION_MINOR)
EXTRA_CFLAGS += -D__IPC__=$(IPC)

ifeq ($(ONE_PIPE),y)
	DEBFLAGS += -DONE_PIPE
endif

EXTRA_CFLAGS += $(DEBFLAGS)
EXTRA_CFLAGS += -std=gnu99 -Wno-declaration-after-statement -fgnu89-inline -D__KERNEL__

TARGET = test_netlink
ccp-objs := test_netlink.o ../src/serialize.o

obj-m := $(TARGET).o

all:
	$(MAKE) -C /lib/modules/$(shell uname -r)/build M=$(CURDIR) modules

clean:
	$(MAKE) -C /lib/modules/$(shell uname -r)/build M=$(CURDIR) clean
