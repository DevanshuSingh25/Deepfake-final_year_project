import { Tab } from '@headlessui/react';
import { cn } from '@/lib/utils';
import { Fragment, ReactNode } from 'react';

interface TabItem {
  label: string;
  content: ReactNode;
}

interface TabsProps {
  tabs: TabItem[];
  className?: string;
}

export function Tabs({ tabs, className }: TabsProps) {
  return (
    <Tab.Group as="div" className={className}>
      <Tab.List className="flex space-x-1 rounded-xl bg-muted p-1">
        {tabs.map((tab) => (
          <Tab key={tab.label} as={Fragment}>
            {({ selected }) => (
              <button
                className={cn(
                  'w-full rounded-lg py-2.5 text-sm font-medium leading-5 transition-all focus-ring',
                  selected
                    ? 'bg-background text-primary shadow'
                    : 'text-muted-foreground hover:bg-background/50 hover:text-foreground'
                )}
              >
                {tab.label}
              </button>
            )}
          </Tab>
        ))}
      </Tab.List>
      <Tab.Panels className="mt-4">
        {tabs.map((tab, idx) => (
          <Tab.Panel
            key={idx}
            className="rounded-xl focus-ring p-3 animate-fade-in"
          >
            {tab.content}
          </Tab.Panel>
        ))}
      </Tab.Panels>
    </Tab.Group>
  );
}
