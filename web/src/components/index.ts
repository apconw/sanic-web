// Vue Component Install

import IconFont from '@/components/IconFont/index.vue'

const Components = {
  install(app: App<any>) {
    componentList.forEach((Comp) => {
      app.component(Comp.name, Comp)
    })
  },
}

const componentList = [
  IconFont,
]

export default Components
